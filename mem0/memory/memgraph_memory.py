import logging
from datetime import datetime, timezone
from mem0.memory.utils import format_entities
from openai import OpenAI
from neo4j import GraphDatabase
import types    
import os
import json
import re



try:
    from langchain_memgraph.graphs.memgraph import Memgraph
except ImportError:
    raise ImportError("langchain_memgraph is not installed. Please install it using pip install langchain-memgraph")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        """Patched initialization that handles existing indexes gracefully"""
        self.config = config
        # Import Memgraph here to avoid import issues
        try:
            from langchain_memgraph.graphs.memgraph import Memgraph
        except ImportError:
            raise ImportError("langchain_memgraph is not installed. Please install it using pip install langchain-memgraph")
        self.graph = Memgraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )
        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = -1
        # Setup Memgraph with error handling:
        # 1. Create vector index (created Entity label on all nodes)
        # 2. Create label property index for performance optimizations
        embedding_dims = self.config.embedder.config["embedding_dims"]
        create_vector_index_query = f"CREATE VECTOR INDEX memzero ON :Entity(embedding) WITH CONFIG {{'dimension': {embedding_dims}, 'capacity': 1000, 'metric': 'cos'}};"
        try:
            self.graph.query(create_vector_index_query, params={})
            logger.info("Successfully created vector index 'memzero'")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Vector index 'memzero' already exists, skipping creation")
            else:
                logger.error(f"Error creating vector index: {e}")
                raise e
        try:
            create_label_prop_index_query = "CREATE INDEX ON :Entity(user_id);"
            self.graph.query(create_label_prop_index_query, params={})
            logger.info("Successfully created label property index on :Entity(user_id)")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Label property index on :Entity(user_id) already exists, skipping creation")
            else:
                logger.error(f"Error creating label property index: {e}")
                raise e
        try:
            create_label_index_query = "CREATE INDEX ON :Entity;"
            self.graph.query(create_label_index_query, params={})
            logger.info("Successfully created label index on :Entity")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Label index on :Entity already exists, skipping creation")
            else:
                logger.error(f"Error creating label index: {e}")
                raise e

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        
        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support

        added_entities = self._add_entities(filters,data)

        return {"added_entities": added_entities}

    def search(self, query, filters, user_id, top_k=5, limit=5):
        print(":gear:  Using Patient-centered entity-based vector search...")
        
        try:
            # Step 1: Extract entities from the query using LLM
            entities = self.extract_entities_with_llm(query, user_id)
            print(f":white_check_mark: User id: {user_id}")
            if not entities:
                print(":warning: No entities extracted, falling back to full query search")
                entities = [query.lower()]
            
            # Step 2: Get embeddings for each entity
            entity_embeddings = {}
            for entity in entities:
                try:
                    entity_embeddings[entity] = self.embedding_model.embed(entity)
                except Exception as e:
                    print(f":warning: Failed to embed entity '{entity}': {e}")
                    continue
            
            if not entity_embeddings:
                print(":x: No valid entity embeddings, falling back to text search")
                raise Exception("No entity embeddings available")
            
            # Step 3: Create direct connection to database
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("memgraph", "memgraph"))
            
            with driver.session() as session:
                # Start from Patient node and traverse relationships to find related nodes
                cypher = """
                MATCH (patient:Patient {user_id: $user_id})-[r]-(related_node)
                WHERE related_node.embedding IS NOT NULL
                RETURN related_node.embedding_text AS text,
                       related_node.user_id AS user_id,
                       related_node.embedding AS embedding,
                       labels(related_node) AS node_type,
                       related_node.name AS node_name,
                       id(related_node) AS node_id,
                       type(r) AS relationship_type
                """
                result = session.run(cypher, parameters={"user_id": user_id})
                
                # Step 4: Calculate entity-based similarities
                node_scores = {}
                
                for record in result:
                    node_id = record["node_id"]
                    node_embedding = record["embedding"]
                    node_text = (record["text"] or "").lower()  # Handle None values
                    
                    # Skip nodes with no text or embedding
                    if not node_text or not node_embedding:
                        continue
                    
                    # Calculate similarity for each entity against this node
                    max_entity_score = 0.0
                    best_matching_entity = None
                    
                    for entity, entity_embedding in entity_embeddings.items():
                        # Calculate cosine similarity between entity and node
                        similarity = self.calculate_cosine_similarity(entity_embedding, node_embedding)
                        
                        # Boost score if entity appears in node text
                        text_boost = 1.0
                        if entity in node_text:
                            text_boost = 1.2
                            print(f":mag: Text match boost for entity '{entity}' in node")
                        
                        final_score = similarity * text_boost
                        
                        if final_score > max_entity_score:
                            max_entity_score = final_score
                            best_matching_entity = entity
                    
                    # Store the best score for this node
                    if max_entity_score > 0:
                        node_scores[node_id] = {
                            "text": record["text"],
                            "score": max_entity_score,
                            "node_type": record["node_type"][0] if record["node_type"] else "Unknown",
                            "node_name": record["node_name"],
                            "matching_entity": best_matching_entity,
                            "relationship_type": record["relationship_type"]
                        }
                
                # Step 5: Sort and return top results
                matches = list(node_scores.values())
                matches.sort(key=lambda x: x["score"], reverse=True)
                matches = matches[:top_k]
                
                print(f":white_check_mark: Found {len(matches)} Patient-related matches")
                for i, match in enumerate(matches[:3], 1):  # Show top 3 for debugging
                    print(f"  {i}. Score: {match['score']:.3f} | Rel: {match['relationship_type']} | Entity: {match['matching_entity']} | Text: {match['text'][:100]}...")
                
                driver.close()
                return matches
                
        except Exception as e:
            print(f":warning:  Entity-based vector search error: {e}")
            # Fallback to Patient-centered entity-based text search
            print(":arrows_counterclockwise: Falling back to Patient-centered text search...")
            
            try:
                # Try to extract entities for text search too
                entities = self.extract_entities_with_llm(query, user_id)
                if not entities:
                    entities = [query.lower()]
                
                driver = GraphDatabase.driver("bolt://localhost:7687", auth=("memgraph", "memgraph"))
                with driver.session() as session:
                    all_matches = []
                    
                    # Search for each entity in Patient's related nodes
                    for entity in entities:
                        cypher = """
                        MATCH (patient:Patient {user_id: $user_id})-[r]-(related_node)
                        WHERE related_node.embedding_text IS NOT NULL
                        AND toLower(related_node.embedding_text) CONTAINS toLower($entity)
                        RETURN related_node.embedding_text AS text,
                               labels(related_node) AS node_type,
                               related_node.name AS node_name,
                               0.6 AS similarity,
                               $entity AS matching_entity,
                               id(related_node) AS node_id,
                               type(r) AS relationship_type
                        """
                        result = session.run(cypher, parameters={
                            "entity": entity,
                            "user_id": user_id
                        })
                        
                        for record in result:
                            all_matches.append({
                                "text": record["text"],
                                "score": record["similarity"],
                                "node_type": record["node_type"][0] if record["node_type"] else "Unknown",
                                "node_name": record["node_name"],
                                "matching_entity": record["matching_entity"],
                                "node_id": record["node_id"],
                                "relationship_type": record["relationship_type"]
                            })
                    
                    # Remove duplicates based on node_id and take top_k
                    unique_matches = {}
                    for match in all_matches:
                        node_id = match["node_id"]
                        if node_id not in unique_matches or match["score"] > unique_matches[node_id]["score"]:
                            unique_matches[node_id] = match
                    
                    matches = list(unique_matches.values())
                    matches.sort(key=lambda x: x["score"], reverse=True)
                    matches = matches[:top_k]
                    
                    print(f":white_check_mark: Found {len(matches)} matches with Patient-centered text search")
                    driver.close()
                    return matches
                    
            except Exception as fallback_error:
                print(f":x: Entity-based fallback search also failed: {fallback_error}")
                return []


        return search_results

    def delete_all(self, filters):
        """Delete all nodes and relationships for a user or specific agent."""
        if filters.get("agent_id"):
            cypher = """
            MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"]}
        else:
            cypher = """
            MATCH (n:Entity {user_id: $user_id})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
                Supports 'user_id' (required) and 'agent_id' (optional).
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'source': The source node name.
                - 'relationship': The relationship type.
                - 'target': The target node name.
        """
        # Build query based on whether agent_id is provided
        if filters.get("agent_id"):
            query = """
            MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})-[r]->(m:Entity {user_id: $user_id, agent_id: $agent_id})
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"], "limit": limit}
        else:
            query = """
            MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity {user_id: $user_id})
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """
            params = {"user_id": filters["user_id"], "limit": limit}

        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results


    def extract_entities_with_llm(self, query, user_id):
        """Extract relevant entities from the query using OpenAI GPT"""
        print(f":brain: Extracting entities from query: '{query}'")
        
        # Enhanced medical prompt based on your search_mem0.py example
        enhanced_prompt = f"""You are a medical data assistant specialized in extracting entities from healthcare queries.
        IMPORTANT RULES:
        1. If the query asks about MEDICATIONS/DRUGS/PRESCRIPTIONS, always include "medications" as an entity
        2. If the query asks about SYMPTOMS/PAIN/CONDITIONS, always include "symptoms" as an entity  
        3. If the query asks about MEALS/FOOD/DIET, always include "meals" as an entity
        4. If the query asks about OBSERVATIONS/TESTS/RESULTS/VITALS, always include "observations" as an entity
        5. If the query asks about ALLERGIES, always include "allergies" as an entity
        6. If the query asks about DIAGNOSES/CONDITIONS, always include "diagnoses" as an entity
        7. If user message contains self reference such as 'I', 'me', 'my' etc. then use {user_id} as the source entity
        8. Extract BOTH the target entity (what user is asking about) AND the subject entity (who it relates to)

        EXAMPLES:
        - "What medications is the patient taking?" → Extract: ["medications", "patient"]
        - "Show me my symptoms" → Extract: ["symptoms"]  
        - "Tell me about my meals" → Extract: ["meals"]
        - "What are the patient's test results?" → Extract: ["observations", "patient"]
        - "knee pain" → Extract: ["symptoms", "knee_pain"]
        - "diabetes medication" → Extract: ["medications", "diabetes"]
        - "blood pressure readings" → Extract: ["observations", "blood_pressure"]
        - "Tell me about my right knee pain" → Extract: ["symptoms", "right_knee_pain"]


        Extract all relevant entities from the text. Return ONLY a JSON array of strings.
        DO NOT answer the question itself. DO NOT include explanations.

        Query: "{query}"

        Entities:"""

        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert medical entity extraction system. Return only JSON arrays."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            print(f":mag: LLM Response: {content}")
            
            # Try to parse as JSON
            try:
                entities = json.loads(content)
                if isinstance(entities, list):
                    entities = [str(entity).strip().lower() for entity in entities if entity]
                    print(f":white_check_mark: Extracted entities: {entities}")
                    return entities
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract using regex if JSON parsing fails
            matches = re.findall(r'"([^"]+)"', content)
            if matches:
                entities = [entity.strip().lower() for entity in matches]
                print(f":arrows_counterclockwise: Regex extracted entities: {entities}")
                return entities
            
            # Final fallback: use original query terms
            entities = [word.strip().lower() for word in query.split() if len(word) > 2]
            print(f":warning: Using fallback word extraction: {entities}")
            return entities
            
        except Exception as e:
            print(f":x: Entity extraction error: {e}")
            # Fallback to simple word extraction
            entities = [word.strip().lower() for word in query.split() if len(word) > 2]
            print(f":arrows_counterclockwise: Using simple word fallback: {entities}")
            return entities

    def calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        # Convert to numpy arrays if they aren't already
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
        
    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Eshtablish relations among the extracted nodes."""
        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]).replace(
                        "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
                    ),
                },
                {"role": "user", "content": data},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]),
                },
                {
                    "role": "user",
                    "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities["tool_calls"]:
            entities = extracted_entities["tool_calls"][0]["arguments"]["entities"]

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []

            
            # Build query based on whether agent_id is provided
            

            #todo Restore original qiuuery after patient embeddings
        cypher_query = """
                MATCH (n:Patient {uuid: $user_id})-[r]->(m)
                WHERE m.embedding IS NOT NULL
                RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship, id(r) AS relation_id,
                    n.name AS destination, id(n) AS destination_id
            """

        ans = self.graph.query(cypher_query, params={"user_id": filters["user_id"]})
        result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)
        system_prompt, user_prompt = get_delete_messages(search_output_string, data, filters["user_id"])

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )
        to_be_deleted = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_deleted.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query
            agent_filter = ""
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
                params["agent_id"] = agent_id

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n:Entity {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m:Entity {{name: $dest_name, user_id: $user_id}})
            WHERE 1=1 {agent_filter}
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    # added Entity label to all nodes for vector search to work
    
    def _add_entities(self, filters, tool_data):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]

        current_time = datetime.now(timezone.utc).isoformat(timespec='microseconds') + 'Z'

        # Find the patient node
        source_result = self.graph.query("""
            MATCH (p:Patient {uuid: $uuid})
            RETURN id(p) AS pid
        """, {"uuid": user_id})

        source_id=None
        if not source_result:
            # Create the patient node with uuid
            create_result = self.graph.query("""
                CREATE (p:Patient:Entity {uuid: $uuid , createdAt: $createdAt})
                RETURN id(p) AS pid
            """, {"uuid": user_id,"createdAt": current_time})
            source_id = create_result[0]["pid"]
        else:        
            source_id = source_result[0]["pid"]
        
        tool = tool_data.get("tool")
        if tool == "add_symptom":
            params, cypher = add_symptom(self, source_id, tool_data)

        elif tool == "add_meal":
            params, cypher = add_meal(self, source_id, tool_data)

        elif tool_data.get("tool") == "add_treatment":
            params, cypher = add_treatment(self, source_id, tool_data)
        else:
            raise ValueError(f"Unsupported tool: {tool}")
        
        params["createdAt"] = current_time
        result = self.graph.query(cypher, params=params)
        return result

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        """Search for source nodes with similar embeddings."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)

        if agent_id:
            cypher = """
                CALL vector_search.search("memzero", 1, $source_embedding) 
                YIELD distance, node, similarity
                WITH node AS source_candidate, similarity
                WHERE source_candidate.user_id = $user_id 
                AND source_candidate.agent_id = $agent_id 
                AND similarity >= $threshold
                RETURN id(source_candidate);
                """
            params = {
                "source_embedding": source_embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "threshold": threshold,
            }
        else:
            cypher = """
                CALL vector_search.search("memzero", 1, $source_embedding) 
                YIELD distance, node, similarity
                WITH node AS source_candidate, similarity
                WHERE source_candidate.user_id = $user_id 
                AND similarity >= $threshold
                RETURN id(source_candidate);
                """
            params = {
                "source_embedding": source_embedding,
                "user_id": user_id,
                "threshold": threshold,
            }

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        """Search for destination nodes with similar embeddings."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)

        if agent_id:
            cypher = """
                CALL vector_search.search("memzero", 1, $destination_embedding) 
                YIELD distance, node, similarity
                WITH node AS destination_candidate, similarity
                WHERE node.user_id = $user_id 
                AND node.agent_id = $agent_id 
                AND similarity >= $threshold
                RETURN id(destination_candidate);
                """
            params = {
                "destination_embedding": destination_embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "threshold": threshold,
            }
        else:
            cypher = """
                CALL vector_search.search("memzero", 1, $destination_embedding) 
                YIELD distance, node, similarity
                WITH node AS destination_candidate, similarity
                WHERE node.user_id = $user_id 
                AND similarity >= $threshold
                RETURN id(destination_candidate);
                """
            params = {
                "destination_embedding": destination_embedding,
                "user_id": user_id,
                "threshold": threshold,
            }

        result = self.graph.query(cypher, params=params)
        return result
    
    def _fetch_existing_indexes(self):
        """
        Retrieves information about existing indexes and vector indexes in the Memgraph database.

        Returns:
            dict: A dictionary containing lists of existing indexes and vector indexes.
        """
        
        index_exists = list(self.graph.query("SHOW INDEX INFO;"))
        vector_index_exists = list(self.graph.query("SHOW VECTOR INDEX INFO;"))
        return {
            "index_exists": index_exists,
            "vector_index_exists": vector_index_exists
        }

def add_symptom(self,source_id,tool_data):
    symptom = tool_data["toolInput"]
    embedding = self.embedding_model.embed(str(symptom))
    cypher = f"""       
        MATCH (p:Patient)
        WHERE id(p) = $source_id
        MERGE (s:Symptom {{
            name: $name,
            severity: $severity,
            duration: $duration,
            relievingFactors: $relievingFactors,
            triggers: $triggers,
            treatmentAndMedication: $treatmentAndMedication
        }})
        SET s.createdAt = $createdAt, s.embedding = $embedding
        CREATE (p)-[r:EXPERIENCES]->(s)
        SET r.createdAt = $createdAt
        RETURN p.uuid AS patient_uuid, type(r) AS relationship, s.name AS symptom
    """
    params = {
        "source_id": source_id,
        "name": symptom["name"],
        "severity": symptom["severity"],
        "duration": symptom["duration"],
        "relievingFactors": symptom["relievingFactors"],
        "triggers": symptom["triggers"],
        "treatmentAndMedication": symptom["treatmentAndMedication"],
        "embedding": embedding,
    }
    return params,cypher
   
def add_meal(self,source_id,tool_data):
    food = tool_data["toolInput"]
    embedding = self.embedding_model.embed(str(food))
    cypher = f"""
        MATCH (p:Patient)
        WHERE id(p) = $source_id
        CREATE (m:Meal {{
            name: $name,
            location: $location,
            ateWith: $ateWith,
            quality: $quality,
            finished: $finished,
            additions: $additions,
            score: $score,
            calories: $calories
        }})
        SET m.createdAt = $createdAt, m.embedding = $embedding
        CREATE (p)-[r:HAS_MEAL]->(m)
        SET r.createdAt = $createdAt
        RETURN p.uuid AS patient_uuid, type(r) AS relationship, m.name AS meal
        """
    params = {
        "source_id": source_id,
        "name": food["name"],
        "location": food["location"],
        "ateWith": food["ateWith"],
        "quality": food["quality"],
        "finished": food["finished"],
        "additions": food.get("additions", ""),
        "score": food.get("score", 0),
        "calories": food.get("calories", 0),
        "embedding": embedding,
    }
    return params, cypher

def add_treatment(self,source_id,tool_data):
    treatment = tool_data["toolInput"]
    embedding = self.embedding_model.embed(str(treatment))
    cypher = """
        MATCH (p:Patient)
        WHERE id(p) = $source_id
        MERGE (t:Treatment{
            name: $name,
            startDate: $startDate,
            endDate: $endDate,
            location: $location,
            sideEffects: $sideEffects,
            notes: $notes
        })
        SET t.createdAt = $createdAt, t.embedding = $embedding
        CREATE (p)-[r:RECEIVED_TREATMENT]->(t)
        SET r.createdAt = $createdAt
        RETURN p.uuid AS patient_uuid, type(r) AS relationship, t.name AS treatment
        """
    params = {
        "source_id": source_id,
        "name": treatment["name"],
        "startDate": treatment["startDate"],
        "endDate": treatment["endDate"],
        "location": treatment["location"],
        "sideEffects": treatment["sideEffects"],
        "notes": treatment["notes"],
        "embedding": embedding,
    }        
    return params, cypher 
