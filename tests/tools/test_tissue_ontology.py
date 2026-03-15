import os
import json
import shutil
import pytest
import tempfile
import networkx as nx
from unittest.mock import patch, MagicMock, mock_open
import appdirs

from SRAgent.tools.tissue_ontology import (
    get_neighbors,
    query_vector_db,
    query_uberon_ols,
    get_uberon_ontology_graph,
    load_vector_store
)

# Fixture to mock appdirs.user_cache_dir to return a temp directory
@pytest.fixture
def mock_cache_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('appdirs.user_cache_dir', return_value=temp_dir):
            yield temp_dir


# Fixture to create a mock OBO graph
@pytest.fixture
def mock_obo_graph():
    # Create a simple mock graph with a few UBERON nodes
    g = nx.MultiDiGraph()
    
    # Add some test nodes
    test_nodes = {
        "UBERON:0000010": {
            "name": "peripheral nervous system",
            "def": "The nervous system outside of the brain and spinal cord."
        },
        "UBERON:0000955": {
            "name": "brain", 
            "def": "The brain is the center of the nervous system in all vertebrate."
        },
        "UBERON:0000956": {
            "name": "neural tissue",
            "def": "Tissue composed of neurons and supporting cells."
        }
    }
    
    # Add nodes to the graph
    for node_id, attrs in test_nodes.items():
        g.add_node(node_id, **attrs)
    
    # Add edges between nodes
    g.add_edge("UBERON:0000010", "UBERON:0000956")
    g.add_edge("UBERON:0000955", "UBERON:0000956")
    
    return g


# Mock for get_ontology_graph function
@pytest.fixture
def mock_get_ontology_graph(mock_obo_graph):
    with patch('SRAgent.tools.tissue_ontology.get_uberon_ontology_graph', return_value=mock_obo_graph):
        yield


# Mock for requests.get
@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b"Mock OBO file content"
        mock_response.json.return_value = {
            "response": {
                "docs": [
                    {
                        "obo_id": "UBERON:0000955",
                        "label": "brain",
                        "description": ["The brain is the center of the nervous system in all vertebrate."]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        yield mock_get


# Mock for chromadb and langchain components
@pytest.fixture
def mock_chroma():
    with patch('SRAgent.tools.vector_db.chromadb.PersistentClient') as mock_client, \
         patch('SRAgent.tools.vector_db.Chroma') as mock_chroma, \
         patch('SRAgent.tools.vector_db.OpenAIEmbeddings') as mock_embeddings, \
         patch('SRAgent.tools.vector_db.AzureOpenAIEmbeddings') as mock_azure_embeddings:
        
        # Configure mock chroma collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_client.return_value.get_collection.return_value = mock_collection
        
        # Configure mock vector store and search results
        mock_result = MagicMock()
        mock_result.metadata = {"id": "UBERON:0000955", "name": "brain"}
        mock_result.page_content = "The brain is the center of the nervous system in all vertebrate."
        
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [mock_result]
        mock_chroma.return_value = mock_vector_store
        
        yield mock_chroma


@patch('os.path.exists', return_value=True)
def test_load_vector_store_with_azure_embeddings(mock_exists):
    """Test load_vector_store with Azure OpenAI embeddings."""
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://example-resource.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
        },
        clear=False,
    ):
        with patch('SRAgent.tools.vector_db.chromadb.PersistentClient') as mock_client, \
             patch('SRAgent.tools.vector_db.Chroma') as mock_chroma, \
             patch('SRAgent.tools.vector_db.AzureOpenAIEmbeddings') as mock_embeddings:

            mock_collection = MagicMock()
            mock_collection.count.return_value = 10
            mock_client.return_value.get_collection.return_value = mock_collection

            load_vector_store('/tmp/chroma', collection_name='uberon')

            mock_embeddings.assert_called_once_with(
                model='text-embedding-3-small',
                azure_deployment='text-embedding-3-small',
                azure_endpoint='https://example-resource.openai.azure.com/',
                api_key='test-key',
                api_version='2024-12-01-preview',
            )


# Mock for tarfile
@pytest.fixture
def mock_tarfile():
    with patch('tarfile.open') as mock_tar:
        mock_tar_instance = MagicMock()
        mock_tar.return_value.__enter__.return_value = mock_tar_instance
        yield mock_tar


# Test get_neighbors function
def test_get_neighbors_invalid_id():
    """Test get_neighbors with invalid Uberon ID format"""
    result = get_neighbors.invoke({"uberon_id": "invalid"})
    assert "Invalid Uberon ID format" in result


@patch('os.path.exists', return_value=True)
def test_get_neighbors_valid_id(mock_exists, mock_get_ontology_graph):
    """Test get_neighbors with valid Uberon ID"""
    result = get_neighbors.invoke({"uberon_id": "UBERON:0000010"})
    assert "Neighbors in the ontology for: \"UBERON:0000010\"" in result
    assert "UBERON:0000956" in result
    assert "neural tissue" in result


@patch('os.path.exists', return_value=False)
def test_get_neighbors_download_obo(mock_exists, mock_requests_get, mock_cache_dir):
    """Test get_neighbors when OBO file needs to be downloaded"""
    with patch('os.makedirs'):
        with patch('builtins.open', mock_open()):
            with patch('SRAgent.tools.tissue_ontology.get_uberon_ontology_graph') as mock_graph:
                # Configure the mock graph to return an empty graph (to avoid processing neighbors)
                mock_graph.return_value = nx.MultiDiGraph()
                
                result = get_neighbors.invoke({"uberon_id": "UBERON:0000010"})
                
                # Verify the download was attempted
                mock_requests_get.assert_called_once()
                assert "http://purl.obolibrary.org/obo/uberon/uberon-full.obo" in mock_requests_get.call_args[0][0]


# Test query_vector_db function
@patch('os.path.exists', return_value=True)
@patch('os.listdir', return_value=['some_file'])
def test_query_vector_db_with_existing_db(mock_listdir, mock_exists, mock_chroma):
    """Test query_vector_db when the Chroma DB already exists"""
    result = query_vector_db.invoke({"query": "brain", "k": 3})
    
    assert "Results for query: \"brain\"" in result
    assert "UBERON:0000955" in result
    assert "brain" in result


def test_query_vector_db_download_db(mock_requests_get, mock_tarfile, mock_cache_dir, mock_chroma):
    """Test query_vector_db when the Chroma DB needs to be downloaded"""
    # Set up a more comprehensive patching strategy
    with patch('os.makedirs'):
        with patch('os.path.exists') as mock_exists:
            # First it checks cache dir, then DB dir, then again for load_vector_store
            mock_exists.side_effect = [True, False, True]
            
            with patch('os.listdir') as mock_listdir:
                # First check is empty, then after extraction it has content
                mock_listdir.side_effect = [[], ['uberon-full_chroma']]
                
                with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
                    mock_temp_dir.return_value.__enter__.return_value = '/tmp/mocktemp'
                    
                    with patch('os.path.isdir', return_value=True):
                        with patch('shutil.move'):
                            with patch('os.remove'):
                                # Call the function
                                result = query_vector_db.invoke({"query": "brain", "k": 3})
                                
                                # Verify the download was attempted
                                mock_requests_get.assert_called_once()
                                assert "storage.googleapis.com" in mock_requests_get.call_args[0][0]
                                
                                # Check the result contains expected text
                                assert "Results for query: \"brain\"" in result


# Test query_uberon_ols function
def test_query_uberon_ols(mock_requests_get):
    """Test query_uberon_ols function"""
    result = query_uberon_ols.invoke({"search_term": "brain"})
    
    # Check that the API was called with the correct URL
    mock_requests_get.assert_called_once()
    assert "ebi.ac.uk/ols/api/search" in mock_requests_get.call_args[0][0]
    assert "brain" in mock_requests_get.call_args[0][0]
    
    # Check the result
    assert "Results from OLS for 'brain'" in result
    assert "UBERON:0000955" in result
    assert "brain" in result


def test_query_uberon_ols_error_handling():
    """Test query_uberon_ols error handling"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("API error")
        
        result = query_uberon_ols.invoke({"search_term": "brain"})
        assert "Error querying OLS API" in result


def test_query_uberon_ols_no_results(mock_requests_get):
    """Test query_uberon_ols when no results are found"""
    mock_requests_get.return_value.json.return_value = {"response": {"docs": []}}
    
    result = query_uberon_ols.invoke({"search_term": "nonexistent_tissue"})
    assert "No results found for search term" in result
