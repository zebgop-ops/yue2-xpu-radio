"""Batched YuE2 decoding: several independent songs share one XPU/CUDA graph per decode step.

The decode step is memory-bandwidth bound, so a second song costs ~4 % and four songs roughly double a card's
throughput (2x Arc Pro B70, 90 s songs: 1.40 -> 3.10 s of music per second at B=4). Each song keeps its own
prompt, seed, sampling history, length cap and stop token; a finished song idles until the batch is done.
With one song this reproduces yue2.sampling.generate_tokens token for token.
"""
import inspect
import time

import torch
import yue2.cuda_graph as cg
from yue2.protocol import ABC_END, MUSIC_END, CONTEXT
from yue2.sampling import distribution, synchronize

# GraphAR is written for one request or a CFG pair; lift the branch limit and add a per-branch token step.
_src = inspect.getsource(cg.GraphAR).replace("class GraphAR:", "class BatchGraphAR:").replace(
    "if len(prefixes) not in {1, 2}:", "if len(prefixes) < 1:")
exec(compile(_src, cg.__file__, "exec"), cg.__dict__)
BatchGraphAR = cg.BatchGraphAR


def _step_many(self, tokens):
    self.tokens.copy_(tokens.reshape(self.branches, 1))
    if self.graph is not None:
        self.graph.replay()
        result = self.output
    else:
        result = self._decode()
    self.steps += 1
    return result


BatchGraphAR.step_many = _step_many


@torch.inference_mode()
def generate_batch(model, prefixes, sampling, seeds, phase, caps=None, cancelled=None, progress=None):
    """Returns [(token_ids, timing, truncated, was_cancelled)] per prefix.
    caps: per-song token limits (<= sampling.max_tokens); cancelled(i) -> bool; progress(i, n_tokens)."""
    device = next(model.parameters()).device
    B = len(prefixes)
    caps = caps or [sampling.max_tokens] * B
    budget = max(caps)
    for p in prefixes:
        if len(p) + budget > CONTEXT:
            raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    gens = [torch.Generator(device=device).manual_seed(s) for s in seeds]
    end = ABC_END if phase == "abc" else MUSIC_END
    synchronize(device); start = time.perf_counter()
    graph = BatchGraphAR(model, prefixes, budget)
    logits = graph.prefill()
    synchronize(device); prefill = time.perf_counter() - start
    hist, done, eos, killed = [[] for _ in range(B)], [False] * B, [False] * B, [False] * B
    end_t = torch.tensor([[end]], device=device)
    step = 0
    try:
        for step in range(budget):
            for i in range(B):
                if not done[i] and cancelled is not None and cancelled(i):
                    done[i] = killed[i] = True
            toks = []
            for i in range(B):
                if done[i]:
                    toks.append(end_t); continue
                scores = distribution(logits[i:i + 1], sampling, hist[i], step, phase)
                toks.append(scores.argmax(-1, keepdim=True) if sampling.temperature == 0 else
                            torch.multinomial(scores.softmax(-1), 1, generator=gens[i]))
            nxt = torch.cat(toks, 0)
            for i, t in enumerate(nxt.view(-1).tolist()):         # one host sync for all songs
                if done[i]:
                    continue
                if t == end:
                    done[i] = eos[i] = True
                else:
                    hist[i].append(t)
                    if len(hist[i]) >= caps[i]:                    # this song's own length cap
                        done[i] = True
            if progress is not None and step % 8 == 0:
                for i in range(B):
                    if not done[i]:
                        progress(i, len(hist[i]))
            if all(done) or step + 1 >= budget:
                break
            logits = graph.step_many(nxt)
        synchronize(device); seconds = time.perf_counter() - start
    finally:
        graph.close()
    total = sum(len(h) + int(e) for h, e in zip(hist, eos))
    out = []
    for i in range(B):
        timing = {"seconds": seconds, "prefill_seconds": prefill, "output_tokens": len(hist[i]) + int(eos[i]),
                  "content_tokens": len(hist[i]), "output_tps": (len(hist[i]) + int(eos[i])) / seconds,
                  "batch": B, "batch_tps": total / seconds, "batch_steps": step + 1, "prefix_tokens": len(prefixes[i]),
                  "cfg_branches": 1, "execution": "cuda_graph" if graph.capture else "eager", "attention": graph.attention_backend}
        out.append((hist[i], timing, not eos[i] and not killed[i], killed[i]))
    return out
