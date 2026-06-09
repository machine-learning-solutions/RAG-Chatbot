/**
 * Netlify Edge — proxy /rag-api/* to ngrok gateway /api/*
 * Copy to portfolio repo: netlify/edge-functions/api-proxy.js
 * Update NGROK_ORIGIN when your tunnel URL changes.
 */
const NGROK_ORIGIN = "https://curable-steerable-obnoxious.ngrok-free.dev";

export default async (request) => {
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/rag-api/, "") || "/";
  const target = new URL(path, NGROK_ORIGIN);
  target.search = url.search;

  const headers = new Headers(request.headers);
  headers.set("Host", new URL(NGROK_ORIGIN).host);
  headers.set("ngrok-skip-browser-warning", "1");
  headers.set("User-Agent", "PortfolioChat/1.0");

  return fetch(target.toString(), {
    method: request.method,
    headers,
    body: request.body,
  });
};
