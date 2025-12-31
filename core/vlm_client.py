"""VLM (Vision Language Model) client wrapper.

For testing purposes, this client reads pre-recorded responses from
planner/capture/response.json instead of making actual API calls.
"""
import os
import json
import time
import threading


class VLMClient:
    """VLM client that uses pre-recorded responses from response.json."""

    # Simulated response delay (seconds)
    RESPONSE_DELAY = 2.0

    def __init__(self, env_id):
        """Initialize VLM client.

        Args:
            env_id: Environment identifier
        """
        self.env_id = env_id
        self.lock = threading.Lock()

        # Load responses from response.json
        self.responses = []
        self.response_index = 0
        self._load_responses()

    def _load_responses(self):
        """Load pre-recorded responses from response.json."""
        response_file = "planner/capture/response.json"

        if not os.path.exists(response_file):
            print(f"[VLMClient] Warning: {response_file} not found!")
            return

        with open(response_file, "r", encoding="utf-8") as f:
            self.responses = json.load(f)

        if not isinstance(self.responses, list):
            print(f"[VLMClient] Warning: response.json should contain a list")
            self.responses = []
            return

        print(f"[VLMClient] Loaded {len(self.responses)} responses from {response_file}")

    def initialize_buffer(self, hash_code):
        """Initialize buffer (no-op for mock, just logs hash).

        Args:
            hash_code: Hash code for cache validation (logged but not used)
        """
        print(f"[VLMClient] initialize_buffer called with hash={hash_code[:32]}...")
        print(f"[VLMClient] Using pre-recorded responses ({len(self.responses)} available)")

    def chat(self, messages, model="gpt-4o"):
        """Return next pre-recorded response with simulated delay.

        Each call independently waits RESPONSE_DELAY seconds before returning.
        Multiple concurrent calls will each wait independently (not queued).

        Args:
            messages: Chat messages (logged for debugging)
            model: Model name (ignored)

        Returns:
            Pre-recorded response string
        """
        # Simulate API latency - sleep BEFORE acquiring lock
        # This allows multiple threads to sleep in parallel
        time.sleep(self.RESPONSE_DELAY)

        # Only lock when accessing shared response index
        with self.lock:
            if self.response_index < len(self.responses):
                response = self.responses[self.response_index]
                self.response_index += 1
                print(f"[VLMClient] chat() returning response {self.response_index}/{len(self.responses)}: {response[:50]}...")
                return response
            else:
                print(f"[VLMClient] ERROR: No more responses! (index={self.response_index})")
                return "ERROR: No more responses available"

    def chat_with_lock(self, messages, model="gpt-4o"):
        """Return next pre-recorded response (same as chat).

        Args:
            messages: Chat messages (logged for debugging)
            model: Model name (ignored)

        Returns:
            Pre-recorded response string
        """
        return self.chat(messages, model)

    def add_capture_to_buffer(self, duration, content):
        """Add capture result to buffer (no-op for mock).

        Args:
            duration: Capture duration
            content: Capture content
        """
        pass

    def reset(self):
        """Reset response index to beginning."""
        self.response_index = 0


class Logger:
    """Logger for LLM interactions."""

    def __init__(self, env_id):
        """Initialize logger.

        Args:
            env_id: Environment identifier
        """
        self.env_id = env_id
        os.makedirs("logs", exist_ok=True)
        self.log_path = f"logs/llm_{env_id}.txt"
        with open(self.log_path, "w") as f:
            f.write("")

    def write(self, *args):
        """Write log entry.

        Args:
            *args: Arguments to log
        """
        log = " ".join([str(arg) for arg in args])
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log + "\n")
        print(log, flush=True)

    def print_role(self, role):
        """Print role header.

        Args:
            role: Role name (System, User, Assistant)
        """
        self.write(f"={role}========================================================")
