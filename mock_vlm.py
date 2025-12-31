"""Mock VLM client for testing with pre-recorded responses.

This module reads VLM responses from capture/response.json file.
The response.json contains a list of pre-recorded VLM responses that
will be returned in order for each chat() call.
"""
import json
import os
import threading


class MockVLMClient:
    """Mock VLM client that returns pre-recorded responses from response.json."""

    def __init__(self, env_id, response_file=None, responses=None):
        """Initialize mock VLM client.

        Args:
            env_id: Environment identifier
            response_file: Path to response.json file (optional)
            responses: List of pre-defined responses (optional)
        """
        self.env_id = env_id
        self.lock = threading.Lock()

        # Response queue
        if responses is not None:
            self.responses = list(responses)
        elif response_file is not None:
            self.responses = self._load_responses(response_file)
        else:
            self.responses = []

        self.response_index = 0
        self.call_history = []

    def _load_responses(self, response_file):
        """Load responses from JSON file.

        Args:
            response_file: Path to response.json

        Returns:
            List of response strings
        """
        if not os.path.exists(response_file):
            print(f"[MockVLM] Warning: Response file not found: {response_file}")
            return []

        with open(response_file, "r", encoding="utf-8") as f:
            responses = json.load(f)

        if not isinstance(responses, list):
            print(f"[MockVLM] Warning: response.json should contain a list")
            return []

        print(f"[MockVLM] Loaded {len(responses)} responses from {response_file}")
        return responses

    def chat(self, messages, model="gpt-4o"):
        """Return next pre-recorded response.

        Args:
            messages: Chat messages (logged for debugging)
            model: Model name (ignored)

        Returns:
            Pre-recorded response string
        """
        # Log the call
        self.call_history.append({
            "method": "chat",
            "messages": messages,
            "index": self.response_index,
        })

        # Return next response
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        else:
            print(f"[MockVLM] Warning: No more responses (index={self.response_index})")
            return ""

    def chat_with_lock(self, messages, model="gpt-4o"):
        """Return next pre-recorded response with thread safety.

        Args:
            messages: Chat messages (logged for debugging)
            model: Model name (ignored)

        Returns:
            Pre-recorded response string
        """
        with self.lock:
            return self.chat(messages, model)

    def reset(self):
        """Reset response index to beginning."""
        self.response_index = 0
        self.call_history = []

    def get_remaining_responses(self):
        """Get count of remaining responses."""
        return len(self.responses) - self.response_index


class MockLogger:
    """Mock logger that captures log output."""

    def __init__(self, env_id, output_file=None):
        """Initialize mock logger.

        Args:
            env_id: Environment identifier
            output_file: Optional file to write logs to
        """
        self.env_id = env_id
        self.output_file = output_file
        self.logs = []

        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w") as f:
                f.write("")

    def write(self, *args):
        """Write log entry."""
        log = " ".join([str(arg) for arg in args])
        self.logs.append(log)

        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log + "\n")

        print(f"[MockLog] {log}", flush=True)

    def print_role(self, role):
        """Print role header."""
        self.write(f"={role}========================================================")

    def get_logs(self):
        """Get all logged entries."""
        return self.logs

    def clear(self):
        """Clear log history."""
        self.logs = []
