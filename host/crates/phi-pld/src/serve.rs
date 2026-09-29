//! The engine over HTTP, shaped like llama-server where the two overlap,
//! so the same request goes to either and the same timing fields come
//! back: `GET /health`, `POST /completion` (a rendered prompt) and `POST
//! /v1/chat/completions` (messages, rendered with the model's template
//! through llama.cpp's own renderer). Greedy only: a `temperature` above 0
//! is refused rather than approximated. One request at a time, as one
//! llama-server slot. See serve.md.

use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use serde_json::{json, Value};

use crate::decode::{generate, Drafter, Outcome, Params};
use crate::llm::Llm;
use crate::lookup::Pick;
use crate::ngram_cache::Caches;

/// A request's drafting: the server's defaults, then the request's own
/// `n_predict` (or `max_tokens`) and `ignore_eos` and, under `"pld"`,
/// any of `k_max`, `k_min`, `adapt`, `min_n`, `max_n`, `fold_max`,
/// `pick` ("first" or "latest"), `drafter` ("exact", "cache" or "both"),
/// `cache_k`, `learn` and `static`.
pub fn params_for(base: &Params, body: &Value) -> Result<Params, String> {
    if body.get("temperature").and_then(Value::as_f64).is_some_and(|t| t > 0.0) {
        return Err("greedy only: temperature above 0 is not supported".into());
    }
    let mut p = base.clone();
    if let Some(n) = body.get("n_predict").or_else(|| body.get("max_tokens")).and_then(Value::as_u64) {
        p.n_predict = n as usize;
    }
    if let Some(v) = body.get("ignore_eos").and_then(Value::as_bool) {
        p.ignore_eos = v;
    }
    if let Some(o) = body.get("pld") {
        let u = |k: &str| o.get(k).and_then(Value::as_u64).map(|v| v as usize);
        if let Some(v) = u("k_max") {
            p.k_max = v;
        }
        if let Some(v) = u("k_min") {
            p.k_min = v;
        }
        if let Some(v) = u("min_n") {
            p.min_n = v;
        }
        if let Some(v) = u("max_n") {
            p.max_n = v;
        }
        if let Some(v) = u("fold_max") {
            p.fold_max = v;
        }
        if let Some(v) = o.get("adapt").and_then(Value::as_bool) {
            p.adapt = v;
        }
        match o.get("pick").and_then(Value::as_str) {
            Some("first") => p.pick = Pick::First,
            Some("latest") => p.pick = Pick::Latest,
            Some(other) => return Err(format!("pick: {other}? first or latest")),
            None => {}
        }
        match o.get("drafter").and_then(Value::as_str) {
            Some("exact") => p.drafter = Drafter::Exact,
            Some("cache") => p.drafter = Drafter::Cache,
            Some("both") => p.drafter = Drafter::Both,
            Some(other) => return Err(format!("drafter: {other}? exact, cache or both")),
            None => {}
        }
        if let Some(v) = u("cache_k") {
            p.cache_k = v;
        }
        if let Some(v) = o.get("learn").and_then(Value::as_bool) {
            p.learn = v;
        }
        if let Some(v) = o.get("static").and_then(Value::as_bool) {
            p.use_static = v;
        }
    }
    Ok(p)
}

/// The timing fields, llama-server's names, and this engine's own.
pub fn timings(o: &Outcome) -> Value {
    let n = o.tokens.len();
    json!({
        "prompt_n": o.prompt_n,
        "prompt_ms": o.prompt_ms,
        "prompt_per_second": o.prompt_n as f64 / (o.prompt_ms / 1e3),
        "predicted_n": n,
        "predicted_ms": o.predicted_ms,
        "predicted_per_second": n as f64 / (o.predicted_ms / 1e3),
        "draft_n": o.draft_n,
        "draft_n_accepted": o.draft_accepted,
        "steps": o.steps,
        "restores": o.restores,
        "carried": o.carried,
        "flushes": o.flushes,
        "by_n": o.by_n.iter().enumerate().filter(|(_, b)| b.steps > 0)
            .map(|(n, b)| json!({"n": n, "steps": b.steps, "drafted": b.drafted, "accepted": b.accepted}))
            .collect::<Vec<_>>(),
        "by_tier": crate::sim::by_tier(o),
        "draft_us_per_token": if n > 0 { o.draft_us / n as f64 } else { 0.0 },
    })
}

fn reply(req: tiny_http::Request, status: u16, body: &Value) {
    let data = body.to_string();
    let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("a valid header");
    let _ = req.respond(tiny_http::Response::from_string(data).with_status_code(status).with_header(header));
}

/// Serve on `addr` until the process is stopped, drafting from `caches`
/// too; when `dynamic` names a file, the dynamic cache is written there
/// after every request.
pub fn serve(llm: Llm, base: Params, mut caches: Caches, dynamic: Option<&Path>, addr: &str) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("{addr}: {e}"))?;
    eprintln!("phi-pld: serving on http://{addr} (POST /completion, POST /v1/chat/completions, GET /health)");
    let llm = Mutex::new(llm);
    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("").to_string();
        let method = req.method().clone();
        if method == tiny_http::Method::Get && path == "/health" {
            reply(req, 200, &json!({"status": "ok"}));
            continue;
        }
        if method != tiny_http::Method::Post || (path != "/completion" && path != "/v1/chat/completions") {
            reply(req, 404, &json!({"error": format!("no {method} {path}")}));
            continue;
        }
        let mut text = String::new();
        if let Err(e) = req.as_reader().read_to_string(&mut text) {
            reply(req, 400, &json!({"error": format!("body: {e}")}));
            continue;
        }
        let body: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                reply(req, 400, &json!({"error": format!("json: {e}")}));
                continue;
            }
        };
        let p = match params_for(&base, &body) {
            Ok(p) => p,
            Err(e) => {
                reply(req, 400, &json!({"error": e}));
                continue;
            }
        };
        let mut llm = llm.lock().unwrap_or_else(|e| e.into_inner());
        let chat = path == "/v1/chat/completions";
        let prompt = if chat {
            let msgs: Vec<(String, String)> = body
                .get("messages")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|m| {
                            let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                            (s("role"), s("content"))
                        })
                        .collect()
                })
                .unwrap_or_default();
            match llm.render_chat(&msgs) {
                Ok(s) => s,
                Err(e) => {
                    reply(req, 400, &json!({"error": format!("{e:#}")}));
                    continue;
                }
            }
        } else {
            body.get("prompt").and_then(Value::as_str).unwrap_or("").to_string()
        };
        let tokens = match llm.tokenize(&prompt) {
            Ok(t) if !t.is_empty() => t,
            Ok(_) => {
                reply(req, 400, &json!({"error": "an empty prompt"}));
                continue;
            }
            Err(e) => {
                reply(req, 400, &json!({"error": format!("{e:#}")}));
                continue;
            }
        };
        let done = generate(&mut *llm, &tokens, &p, &mut caches);
        if let (Some(f), true) = (dynamic, caches.learn) {
            if let Err(e) = caches.dynamic.save(f) {
                eprintln!("phi-pld: the dynamic cache was not written: {e:#}");
            }
        }
        match done {
            Ok(mut o) => {
                o.text = llm.text(&o.tokens);
                let t = timings(&o);
                let body = if chat {
                    json!({"object": "chat.completion",
                           "choices": [{"index": 0, "finish_reason": if o.eog { "stop" } else { "length" },
                                        "message": {"role": "assistant", "content": o.text}}],
                           "timings": t})
                } else {
                    let mut b = json!({"content": o.text, "stop": o.eog, "tokens_predicted": o.tokens.len(), "timings": t});
                    if body.get("return_tokens").and_then(Value::as_bool) == Some(true) {
                        b["tokens"] = json!(o.tokens);
                    }
                    b
                };
                reply(req, 200, &body);
            }
            Err(e) => reply(req, 500, &json!({"error": format!("{e:#}")})),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Params {
        Params {
            n_predict: 100,
            k_max: 10,
            k_min: 1,
            adapt: false,
            min_n: 1,
            max_n: 3,
            pick: Pick::First,
            fold_max: 16,
            junk: false,
            drafter: Drafter::Exact,
            cache_k: 2,
            ignore_eos: false,
            learn: true,
            use_static: true,
        }
    }

    #[test]
    fn a_request_sets_its_own_length_and_drafting() {
        let p = params_for(
            &base(),
            &json!({"n_predict": 7, "pld": {"k_max": 16, "adapt": true, "pick": "latest", "min_n": 2}}),
        )
        .unwrap();
        assert_eq!((p.n_predict, p.k_max, p.adapt, p.pick, p.min_n), (7, 16, true, Pick::Latest, 2));
        let p = params_for(&base(), &json!({"max_tokens": 9})).unwrap();
        assert_eq!(p.n_predict, 9);
        let p = params_for(&base(), &json!({"ignore_eos": true})).unwrap();
        assert!(p.ignore_eos);
    }

    #[test]
    fn sampling_is_refused() {
        assert!(params_for(&base(), &json!({"temperature": 0.7})).is_err());
        assert!(params_for(&base(), &json!({"temperature": 0})).is_ok());
        assert!(params_for(&base(), &json!({"pld": {"pick": "middle"}})).is_err());
    }
}
