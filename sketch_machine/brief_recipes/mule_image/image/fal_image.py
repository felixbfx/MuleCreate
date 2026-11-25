"""Generate or edit one image through fal (GPT Image 2.6 or Nano Banana 2.1), with a spend log and a budget cap.

    python fal_image.py --out out/hero_wide.png --prompt-file prompts/hero_wide.txt --image blockout_wide.png
    python fal_image.py --out out/x.png --prompt "..." --model nano --image a.png --image b.png
    python fal_image.py --spend                 # what has been spent so far (estimates)

FAL_KEY comes from the environment and is never printed or written anywhere.
With --image the edit endpoint is used (the images are the references), without it text-to-image.

Budget: every paid call appends an estimate to spend.jsonl next to this file. A call is refused when the
total would pass the cap (--budget, or FAL_BUDGET_USD, default 20). The estimates are rough upper values;
the fal dashboard has the real numbers.

Re-running is free: an existing <out> is skipped (--force regenerates) and a job that was cut off
(<out>.pending.json) is polled to its end instead of being paid for again.

Standard library only. The queue flow (submit, poll status_url, read response_url) is the public fal queue
API, as also used in apekshik/anymachine hardsurface/tools/detail_fal.mjs (MIT).
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SPEND_LOG = os.path.join(HERE, "spend.jsonl")

# short name -> (edit endpoint, text-to-image endpoint)
MODELS = {
    "gpt": ("openai/gpt-image-2.6/sunburst/edit", "openai/gpt-image-2.6/sunburst/text-to-image"),
    "nano": ("google/nano-banana-2.1/edit", "google/nano-banana-2.1"),
}
# rough upper estimates in USD per image, references included (fal model pages, 2026-11-22)
EST_GPT = {"low": 0.03, "medium": 0.05, "high": 0.07, "xhigh": 0.11, "max": 0.25}
EST_NANO = {"1K": 0.05, "2K": 0.07, "4K": 0.13}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def spent():
    if not os.path.exists(SPEND_LOG):
        return 0.0, 0
    rows = [json.loads(line) for line in open(SPEND_LOG, encoding="utf-8") if line.strip()]
    return sum(r["est_usd"] for r in rows), len(rows)


def as_url(path):
    if path.startswith(("http://", "https://", "data:")):
        return path
    mime = MIME.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def call(url, key, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Key " + key, "Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"fal {e.code} on {url.split('?')[0]}: {e.read().decode(errors='replace')[:600]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out")
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--image", action="append", default=[], help="reference image (path or URL); repeatable")
    p.add_argument("--model", default="gpt", help="gpt, nano, or a full fal endpoint id")
    p.add_argument("--quality", default="high", help="gpt: low, medium, high, xhigh, max")
    p.add_argument("--size", default=None, help="gpt: WxH or a preset such as landscape_16_9 (default: auto with "
                                                "references, landscape_16_9 without)")
    p.add_argument("--resolution", default="2K", help="nano: 1K, 2K, 4K")
    p.add_argument("--aspect", default=None, help="nano: aspect ratio such as 16:9 (default: auto)")
    p.add_argument("--budget", type=float, default=float(os.environ.get("FAL_BUDGET_USD", "20")))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="show what would be sent; no call, no cost")
    p.add_argument("--spend", action="store_true", help="print the spend log total and exit")
    a = p.parse_args()

    total, n = spent()
    if a.spend:
        print(f"{n} calls, about ${total:.2f} estimated (cap ${a.budget:.2f}); log: {SPEND_LOG}")
        return
    if not a.out or not (a.prompt or a.prompt_file):
        p.error("--out and --prompt or --prompt-file are required")
    prompt = a.prompt or open(a.prompt_file, encoding="utf-8").read().strip()
    out = os.path.abspath(a.out)
    pending = out + ".pending.json"
    if os.path.exists(out) and not a.force:
        print(f"exists {out} (skipped; --force to regenerate)")
        return

    edit = bool(a.image)
    if a.model in MODELS:
        endpoint = MODELS[a.model][0 if edit else 1]
    else:
        endpoint = a.model
    is_nano = "nano-banana" in endpoint
    body = {"prompt": prompt, "num_images": 1, "output_format": "png"}
    if is_nano:
        body["resolution"] = a.resolution
        if a.aspect:
            body["aspect_ratio"] = a.aspect
        est = EST_NANO.get(a.resolution, 0.13)
    else:
        body["quality"] = a.quality
        size = a.size or ("auto" if edit else "landscape_16_9")
        if "x" in size and size.replace("x", "").isdigit():
            w, h = size.split("x")
            size = {"width": int(w), "height": int(h)}
        body["image_size"] = size
        est = EST_GPT.get(a.quality, 0.25)
    for path in a.image:
        if not path.startswith(("http://", "https://", "data:")) and not os.path.exists(path):
            raise SystemExit(f"reference image not found: {path}")

    shown = dict(body, image_urls=[os.path.basename(i) for i in a.image]) if edit else body
    if a.dry_run:
        print(f"dry run: {endpoint}\nestimate ${est:.2f}; spent so far ${total:.2f} of ${a.budget:.2f}\n"
              + json.dumps(shown, indent=1, ensure_ascii=False)[:1500])
        return

    key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if not key:
        raise SystemExit("FAL_KEY is not set in the environment")
    t0 = time.time()
    if os.path.exists(pending) and not a.force:
        job = json.load(open(pending, encoding="utf-8"))
        print(f"resuming {endpoint} {job['request_id']}")
    else:
        if total + est > a.budget:
            raise SystemExit(f"budget: about ${total:.2f} spent, this call about ${est:.2f}, cap ${a.budget:.2f}. "
                             "Raise --budget or FAL_BUDGET_USD if that is intended.")
        if edit:
            body["image_urls"] = [as_url(i) for i in a.image]
        job = call("https://queue.fal.run/" + endpoint, key, body)
        if not job or not job.get("request_id"):
            raise SystemExit("submit failed: " + json.dumps(job)[:600])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({k: job[k] for k in ("request_id", "status_url", "response_url") if k in job},
                  open(pending, "w", encoding="utf-8"))
        with open(SPEND_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "endpoint": endpoint, "est_usd": est,
                                "request_id": job["request_id"], "out": os.path.relpath(out, HERE)}) + "\n")
        print(f"queued {endpoint} {job['request_id']} (about ${est:.2f})")

    while True:
        time.sleep(3)
        st = call(job["status_url"], key)
        status = (st or {}).get("status")
        if status == "COMPLETED":
            break
        if status not in ("IN_QUEUE", "IN_PROGRESS"):
            raise SystemExit("job failed: " + json.dumps(st)[:600])
        if time.time() - t0 > 15 * 60:
            raise SystemExit(f"timed out; request {job['request_id']} (re-run to pick it up)")
    res = call(job["response_url"], key)
    images = (res or {}).get("images") or []
    if not images:
        raise SystemExit("no image in the response: " + json.dumps(res)[:600])
    with urllib.request.urlopen(images[0]["url"], timeout=120) as r:
        data = r.read()
    with open(out, "wb") as f:
        f.write(data)
    json.dump({"endpoint": endpoint, "request_id": job["request_id"], "seconds": round(time.time() - t0),
               "input": shown, "url": images[0]["url"]}, open(out + ".json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    os.remove(pending)
    print(f"saved {out} ({len(data) // 1024} KB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
