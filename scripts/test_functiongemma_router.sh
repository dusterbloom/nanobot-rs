#!/bin/bash
# Test FunctionGemma-270M-IT routing capabilities via llama.cpp server
# Port 8094 should have functiongemma loaded

set -e

ROUTER_PORT="${ROUTER_PORT:-8094}"
ROUTER_URL="http://localhost:${ROUTER_PORT}/v1/chat/completions"

# Check server is running
if ! curl -s "http://localhost:${ROUTER_PORT}/v1/models" > /dev/null 2>&1; then
    echo "Error: Router server not running on port ${ROUTER_PORT}"
    echo "Start it with: llama-server -m ~/models/functiongemma-270m-it-Q8_0.gguf --port 8094"
    exit 1
fi

echo "=== FunctionGemma Router Test ==="
echo "Server: ${ROUTER_URL}"
echo ""

# Test 1: Native function calling format
test_native() {
    echo "--- Test 1: Native Function Calling Format ---"
    local response
    response=$(curl -s "${ROUTER_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
                {"role": "user", "content": "Read the file Cargo.toml"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"}
                        },
                        "required": ["path"]
                    }
                }
            }],
            "max_tokens": 128,
            "temperature": 0.1
        }')
    
    echo "Response:"
    echo "${response}" | jq -r '.choices[0].message.content // .choices[0].message.tool_calls // .' 2>/dev/null || echo "${response}"
    echo ""
}

# Test 2: JSON output via few-shot
test_json_fewshot() {
    echo "--- Test 2: JSON Output via Few-Shot ---"
    local response
    response=$(curl -s "${ROUTER_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "You are a router. Output ONLY valid JSON with no explanation. Format: {\"action\":\"tool\",\"target\":\"tool_name\",\"args\":{},\"confidence\":0.9}"},
                {"role": "user", "content": "Read the file Cargo.toml"},
                {"role": "assistant", "content": "{\"action\":\"tool\",\"target\":\"read_file\",\"args\":{\"path\":\"Cargo.toml\"},\"confidence\":0.95}"},
                {"role": "user", "content": "Search the web for gold prices"},
                {"role": "assistant", "content": "{\"action\":\"tool\",\"target\":\"web_search\",\"args\":{\"query\":\"gold prices\"},\"confidence\":0.9}"},
                {"role": "user", "content": "Execute a pipeline to analyze the codebase"}
            ],
            "max_tokens": 64,
            "temperature": 0.0
        }')
    
    echo "Response:"
    echo "${response}" | jq -r '.choices[0].message.content // .' 2>/dev/null || echo "${response}"
    echo ""
}

# Test 3: Classification without tools (pure text)
test_classification() {
    echo "--- Test 3: Classification Task ---"
    local response
    response=$(curl -s "${ROUTER_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "Classify the intent. Respond with one word: tool, specialist, pipeline, or clarify"},
                {"role": "user", "content": "Read the file Cargo.toml"},
                {"role": "assistant", "content": "tool"},
                {"role": "user", "content": "Search the web for gold prices"},
                {"role": "assistant", "content": "tool"},
                {"role": "user", "content": "Analyze the entire codebase and create a comprehensive report"}
            ],
            "max_tokens": 16,
            "temperature": 0.0
        }')
    
    echo "Response:"
    echo "${response}" | jq -r '.choices[0].message.content // .' 2>/dev/null || echo "${response}"
    echo ""
}

# Test 4: Multiple tools available
test_multiple_tools() {
    echo "--- Test 4: Multiple Tools Selection ---"
    local response
    response=$(curl -s "${ROUTER_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
                {"role": "user", "content": "What is the price of gold in London today?"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "exec", "description": "Execute a shell command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}
            ],
            "max_tokens": 128,
            "temperature": 0.1
        }')
    
    echo "Response:"
    echo "${response}" | jq -r '.choices[0].message.content // .choices[0].message.tool_calls // .' 2>/dev/null || echo "${response}"
    echo ""
}

# Test 5: Parallel function calls
test_parallel() {
    echo "--- Test 5: Parallel Function Calls ---"
    local response
    response=$(curl -s "${ROUTER_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
                {"role": "user", "content": "Read both Cargo.toml and Cargo.lock files"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
            ],
            "max_tokens": 256,
            "temperature": 0.1
        }')
    
    echo "Response:"
    echo "${response}" | jq -r '.choices[0].message.content // .choices[0].message.tool_calls // .' 2>/dev/null || echo "${response}"
    echo ""
}

# Run tests
test_native
test_json_fewshot
test_classification
test_multiple_tools
test_parallel

echo "=== Tests Complete ==="
