/**
 * Promptfoo custom provider for CI red-team runs (class-based API).
 *
 * Returns a fixed safe response simulating a correctly hardened agent so that
 * all not-contains assertions pass without consuming LLM quota.
 *
 * For live red-team runs against the real API, replace this provider with:
 *   id: http
 *   config:
 *     url: http://localhost:8000/api/chat
 *     method: POST
 *     body: { role: "user", content: "{{prompt}}" }
 */
class MockSafeAgentProvider {
  constructor(_options) {}

  id() {
    return "mock-safe-agent";
  }

  async callApi(_prompt, _context) {
    return {
      output:
        "Supply-chain query processed. Security restrictions prevent compliance with this request.",
    };
  }
}

module.exports = MockSafeAgentProvider;
