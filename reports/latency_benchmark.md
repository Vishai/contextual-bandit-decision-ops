# Decision Latency Benchmark

Target: `in-process`. 2,000 decide calls, each followed by a
feedback call that updates the posterior online. Contexts come from the SQL
feature store and outcomes from the simulator's true conversion model.

**Result: PASS.** The p99 decide latency is compared with the 100 ms budget.

| Call | Count | p50 ms | p95 ms | p99 ms | Max ms | Within budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| POST /decide (client) | 2,000 | 2.11 | 2.99 | 3.61 | 74.06 | 100.00% |
| POST /decide (server) | 2,000 | 1.34 | 1.96 | 2.47 | 73.21 | 100.00% |
| POST /feedback (client) | 2,000 | 1.78 | 2.54 | 3.17 | 42.98 | 100.00% |

Client timing covers the full request through the HTTP stack. Server timing comes
from the `X-Process-Time-Ms` header that the service middleware sets.

## Online learning during the run

| Half | Conversion | Revenue / decision |
| --- | ---: | ---: |
| First | 9.80% | 4.226 |
| Second | 10.40% | 4.424 |

| Offer | First half | Second half |
| --- | ---: | ---: |
| bundle_deal | 269 | 291 |
| free_shipping | 290 | 271 |
| loyalty_points | 128 | 171 |
| pct10_discount | 115 | 91 |
| premium_trial | 198 | 176 |

The service applied 2,020 posterior updates during the run.
