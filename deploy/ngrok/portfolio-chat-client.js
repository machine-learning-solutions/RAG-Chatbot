/**
 * Portfolio native chat client — SSE streaming via Netlify /rag-api/chat
 *
 * Usage (portfolio site):
 *   <script src="/portfolio-chat-client.js"></script>
 *   const chat = createPortfolioChat({ apiBase: "/rag-api" });
 *   const result = await chat.ask("ما هي مهارات جهاد؟", "ar");
 */
(function (global) {
  function parseSseEvents(buffer) {
    const events = [];
    const parts = buffer.split("\n\n");
    const rest = parts.pop() || "";
    for (const part of parts) {
      if (!part.trim()) continue;
      let event = "message";
      let data = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      events.push({ event, data });
    }
    return { events, rest };
  }

  function createPortfolioChat(options) {
    const apiBase = (options && options.apiBase) || "/rag-api";

    async function ask(question, language) {
      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          question,
          language: language || "ar",
          portfolio_fast: true,
          stream: true,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Chat failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseEvents(buffer);
        buffer = parsed.rest;
        for (const evt of parsed.events) {
          if (evt.event === "error") {
            const err = JSON.parse(evt.data);
            throw new Error(err.detail || "Chat failed");
          }
          if (evt.event === "result") {
            return JSON.parse(evt.data);
          }
        }
      }

      throw new Error("Chat ended without a result");
    }

    return { ask };
  }

  global.createPortfolioChat = createPortfolioChat;
})(typeof window !== "undefined" ? window : globalThis);
