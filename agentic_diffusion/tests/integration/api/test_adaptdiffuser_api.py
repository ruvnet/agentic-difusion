import pytest
import time
import json
from jsonschema import validate

@pytest.mark.integration
class TestAdaptDiffuserAPI:
    """
    Integration tests for the AdaptDiffuser API endpoints.
    
    These tests verify that the API functions correctly by testing:
    1. Basic functionality for all endpoints
    2. Error handling for invalid inputs
    3. Edge cases
    4. Asynchronous operations
    5. Response schema validation
    """
    
    # Schema definitions for response validation
    adapt_response_schema = {
        "type": "object",
        "required": ["adapted_output", "metrics"],
        "properties": {
            "adapted_output": {"type": "string"},
            "metrics": {
                "type": "object",
                "properties": {
                    "iterations": {"type": "number"},
                    "batch_size": {"type": "number"},
                    "adaptation_score": {"type": "number"}
                }
            }
        }
    }
    
    generate_response_schema = {
        "type": "object",
        "required": ["generated_code", "metadata"],
        "properties": {
            "generated_code": {"type": "string"},
            "metadata": {
                "type": "object",
                "properties": {
                    "batch_size": {"type": "number"},
                    "guidance_scale": {"type": "number"},
                    "generation_time_ms": {"type": "number"}
                }
            }
        }
    }
    
    evaluate_response_schema = {
        "type": "object",
        "required": ["score", "details"],
        "properties": {
            "score": {"type": "number"},
            "details": {
                "type": "object",
                "properties": {
                    "quality": {"type": "number"},
                    "efficiency": {"type": "number"},
                    "readability": {"type": "number"},
                    "task": {"type": "string"}
                }
            }
        }
    }

    def test_adapt_diffuser_api_endpoints(self, adaptdiffuser_api_client):
        """
        Comprehensive test for all main API endpoints.
        
        Given: A running AdaptDiffuser API server and test fixtures
        When: API endpoints for adaptation, generation, and evaluation are called
        Then: The responses should match expected status codes and output structure
        """
        # Test 1: Adaptation endpoint with basic parameters
        adaptation_response = adaptdiffuser_api_client.post(
            "/adapt",
            json={"input": "test code", "iterations": 2, "batch_size": 4}
        )
        assert adaptation_response.status_code == 200, "Adaptation endpoint did not return 200"
        adapt_data = adaptation_response.json()
        assert "adapted_output" in adapt_data, "Missing adapted_output in response"
        assert "metrics" in adapt_data, "Missing metrics in response"
        
        # Validate response schema
        validate(instance=adapt_data, schema=self.adapt_response_schema)
        
        # Test 2: Generation endpoint with custom parameters
        generation_response = adaptdiffuser_api_client.post(
            "/generate",
            json={"prompt": "test prompt", "batch_size": 2, "guidance_scale": 7.5}
        )
        assert generation_response.status_code == 200, "Generation endpoint did not return 200"
        gen_data = generation_response.json()
        assert "generated_code" in gen_data, "Missing generated_code in response"
        assert "metadata" in gen_data, "Missing metadata in response"
        
        # Validate response schema
        validate(instance=gen_data, schema=self.generate_response_schema)
        
        # Test 3: Evaluation endpoint with task parameter
        evaluation_response = adaptdiffuser_api_client.post(
            "/evaluate",
            json={"code": "def hello(): print('hi')", "task": "code_quality"}
        )
        assert evaluation_response.status_code == 200, "Evaluation endpoint did not return 200"
        eval_data = evaluation_response.json()
        assert "score" in eval_data, "Missing score in response"
        assert "details" in eval_data, "Missing details in response"
        assert eval_data["details"]["task"] == "code_quality", "Task parameter not reflected in response"
        
        # Validate response schema
        validate(instance=eval_data, schema=self.evaluate_response_schema)

    def test_error_handling(self, adaptdiffuser_api_client):
        """
        Test error handling for invalid inputs and edge cases.
        
        Given: A running AdaptDiffuser API server
        When: Invalid or edge case inputs are provided
        Then: The API should return appropriate error responses
        """
        # Test 1: Missing required parameter (input)
        error_response = adaptdiffuser_api_client.post("/adapt", json={})
        assert error_response.status_code == 422, "Expected 422 for missing required parameter"
        
        # Test 2: Empty input
        empty_response = adaptdiffuser_api_client.post("/adapt", json={"input": ""})
        assert empty_response.status_code == 400, "Expected 400 for empty input"
        
        # Test 3: Empty prompt
        empty_prompt = adaptdiffuser_api_client.post("/generate", json={"prompt": ""})
        assert empty_prompt.status_code == 422, "Expected 422 for empty prompt"
        
        # Test 4: Missing code parameter
        missing_code = adaptdiffuser_api_client.post("/evaluate", json={})
        assert missing_code.status_code == 422, "Expected 422 for missing code parameter"
        
        # Test 5: Invalid JSON
        headers = {"Content-Type": "application/json"}
        invalid_json = adaptdiffuser_api_client.post(
            "/adapt",
            data="not a valid json",
            headers=headers
        )
        assert invalid_json.status_code == 422, "Expected 422 for invalid JSON"

    def test_async_operations(self, adaptdiffuser_api_client):
        """
        Test asynchronous API operations.
        
        Given: A running AdaptDiffuser API server
        When: Async endpoints are called
        Then: Tasks should be created and status should be retrievable
        """
        # Test 1: Start async adaptation task
        async_response = adaptdiffuser_api_client.post(
            "/adapt_async",
            json={"input": "async test code"}
        )
        assert async_response.status_code == 200, "Async endpoint did not return 200"
        async_data = async_response.json()
        assert "task_id" in async_data, "Missing task_id in async response"
        assert "status" in async_data, "Missing status in async response"
        assert async_data["status"] == "processing", "Initial status should be 'processing'"
        
        # Get the task ID for status checking
        task_id = async_data["task_id"]
        
        # Test 2: Check task status (may complete quickly in test environment)
        status_response = adaptdiffuser_api_client.get(f"/adapt_async/{task_id}")
        assert status_response.status_code == 200, "Status endpoint did not return 200"
        status_data = status_response.json()
        assert "status" in status_data, "Missing status in status response"
        
        # Test 3: Wait for task completion (with timeout)
        # Note: In a real environment, you might poll this endpoint
        max_retries = 5
        for _ in range(max_retries):
            status_response = adaptdiffuser_api_client.get(f"/adapt_async/{task_id}")
            status_data = status_response.json()
            if status_data["status"] == "completed":
                assert "result" in status_data, "Missing result in completed task"
                break
            time.sleep(0.5)  # Short wait between checks
        
        # Test 4: Non-existent task
        not_found = adaptdiffuser_api_client.get("/adapt_async/nonexistent-task-id")
        assert not_found.status_code == 404, "Expected 404 for nonexistent task"
        
        # Uncomment to verify the test is red in TDD cycle
        # assert False, "Test skeleton: Failing intentionally to start TDD cycle"