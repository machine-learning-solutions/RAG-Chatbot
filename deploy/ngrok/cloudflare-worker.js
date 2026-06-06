/**
 * Cloudflare Worker — proxy /chat/* to ngrok with skip header
 * Use if jehadabuawwad.com is on Cloudflare (Workers & Routes).
 *
 * Route: jehadabuawwad.com/chat/*
 * Update NGROK_ORIGIN when your tunnel URL changes.
 */
const NGROK_ORIGIN = "https://curable-steerable-obnoxious.ngrok-free.dev";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/chat")) {
      return fetch(request);
    }

    const target = new URL(url.pathname.replace(/^\/chat/, "") || "/", NGROK_ORIGIN);
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
  },
};
