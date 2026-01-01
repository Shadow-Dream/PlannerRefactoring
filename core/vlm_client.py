"""VLM (Vision Language Model) client wrapper.

For testing purposes, this client reads pre-recorded responses from
planner/capture/response.json instead of making actual API calls.

Conditional Response Syntax:
    [IFDONE:X]response_if_done[ELSE]response_else

    This checks if any action in the input is in "(done, Ys)" format
    where Y > X seconds.
    - If TRUE: pop response from queue, return response_if_done
    - If FALSE: do NOT pop (stays for retry), return response_else

    Example:
        [IFDONE:2]stop("sitting on bed1")[ELSE]skip(0.5)
        - If "sitting on bed1 (done, 3.5s)" in input → pops, returns stop()
        - If "sitting on bed1 (acting, 1.0s)" in input → keeps, returns skip()
"""
import os
import json
import time
import re
import threading


class VLMClient:
    """VLM client that uses pre-recorded responses from response.json."""

    # Simulated response delay (seconds)
    RESPONSE_DELAY = 0.5

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

        Supports conditional responses with [IFDONE:X]...[ELSE]... syntax.
        - If condition is TRUE: pop response, return IF branch
        - If condition is FALSE: do NOT pop, return ELSE branch (response stays for retry)

        Args:
            messages: Chat messages (used for conditional evaluation)
            model: Model name (ignored)

        Returns:
            Pre-recorded response string (evaluated if conditional)
        """
        # Simulate API latency - sleep BEFORE acquiring lock
        time.sleep(self.RESPONSE_DELAY)

        # Only lock when accessing shared response index
        with self.lock:
            if self.response_index < len(self.responses):
                raw_response = self.responses[self.response_index]

                # Check if this is a conditional response
                match = re.match(r'\[IFDONE:(\d+(?:\.\d+)?)\](.*?)\[ELSE\](.*)', raw_response, re.DOTALL)
                if match:
                    threshold = float(match.group(1))
                    response_if_done = match.group(2).strip()
                    response_else = match.group(3).strip()

                    # Get the last user message content
                    last_user_content = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            last_user_content = msg.get("content", "")
                            break

                    # Find all actions with (done, Xs) pattern
                    done_pattern = r'\(done,\s*(\d+(?:\.\d+)?)s\)'
                    done_matches = re.findall(done_pattern, last_user_content)

                    # Check if any done action exceeds threshold
                    condition_met = False
                    for duration_str in done_matches:
                        duration = float(duration_str)
                        if duration > threshold:
                            condition_met = True
                            break

                    if condition_met:
                        # Condition TRUE: pop response, return IF branch
                        self.response_index += 1
                        print(f"[VLMClient] Conditional TRUE: popping response, returning: {response_if_done[:50]}...")
                        return response_if_done
                    else:
                        # Condition FALSE: do NOT pop, return ELSE branch
                        print(f"[VLMClient] Conditional FALSE: keeping response, returning: {response_else[:50]}...")
                        return response_else
                else:
                    # Non-conditional: always pop
                    self.response_index += 1
                    print(f"[VLMClient] chat() returning response {self.response_index}/{len(self.responses)}: {raw_response[:50]}...")
                    return raw_response
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
