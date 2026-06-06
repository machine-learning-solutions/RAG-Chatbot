/**
 * Netlify Edge Function — proxy /chat/* to ngrok with skip header
 * For jehadabuawwad.com (hosted on Netlify).
 *
 * 1. Copy to portfolio repo: netlify/edge-functions/chat-proxy.js
 * 2. In netlify.toml:
 *
 *    [[edge_functions]]
 *      path = "/chat/*"
 *      function = "chat-proxy"
 *
 * 3. Set iframe src to: https://jehadabuawwad.com/chat/?portfolio=true
 *
 * Update NGROK_ORIGIN when your tunnel URL changes.
 */
const NGROK_ORIGIN = "https://curable-steerable-obnoxious.ngrok-free.dev";

export default async (request, context) => {
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/chat/, "") || "/";
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
    redirect: "follow",
  });
};
