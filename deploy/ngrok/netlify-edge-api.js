/**
 * Netlify Edge — proxy /rag-api/* → ngrok gateway /api/*
 * Copy to portfolio repo: netlify/edge-functions/api-proxy.js
 *
 * Chat uses SSE streaming so Netlify keeps the connection open (>40s processing).
 * Pair with deploy/ngrok/portfolio-chat-client.js in the portfolio site.
 */
const NGROK_ORIGIN = "https://curable-steerable-obnoxious.ngrok-free.dev";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Accept",
};

export default async (request) => {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS });
  }

  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/rag-api/, "/api") || "/api";
  const target = new URL(path, NGROK_ORIGIN);
  target.search = url.search;

  const headers = new Headers(request.headers);
  headers.set("Host", new URL(NGROK_ORIGIN).host);
  headers.set("ngrok-skip-browser-warning", "1");
  headers.set("User-Agent", "PortfolioChat/1.0");
  headers.delete("content-length");

  const isChatPost =
    request.method === "POST" && path.replace(/\/$/, "") === "/api/chat";

  let body;
  if (request.method !== "GET" && request.method !== "HEAD") {
    const raw = await request.text();
    if (isChatPost && raw) {
      try {
        const payload = JSON.parse(raw);
        payload.stream = true;
        body = JSON.stringify(payload);
      } catch {
        body = raw;
      }
    } else {
      body = raw;
    }
  }

  if (isChatPost) {
    headers.set("Accept", "text/event-stream");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000);

  try {
    const upstream = await fetch(target.toString(), {
      method: request.method,
      headers,
      body,
      redirect: "follow",
      signal: controller.signal,
    });

    if (isChatPost && upstream.ok) {
      return new Response(upstream.body, {
        status: 200,
        headers: {
          ...CORS,
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    return upstream;
  } finally {
    clearTimeout(timeout);
  }
};
