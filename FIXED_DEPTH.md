# Fixed depth at the router's average pair budget

Register before calculating fixed-15 rankings or timings. The primary routers score about 15 candidates per request when averaged over the test set. A constant depth of 15 is a simpler comparator with a similar pair count. It also changes the candidate set differently from a randomized mix of depths 10 and 20. Use this control regardless of the outcome.

Compute the paired difference in query nDCG@10 among fixed 15, fixed 20, and each learned policy on the original official ESCI test. Plot the complete 10–20 fixed-depth curve descriptively. Fixed 15 is the prespecified direct comparator; do not select the best depth on the test. Verify fixed-15 quality with actual single-batch inference on all 1,803 official queries. Repeat the fixed-depth quality curve on the additional previously unevaluated probe when its results become available.

Run a fresh control block on the same Ubuntu worker, paired queries and recorded arrival traces. Include fixed 15, fixed 20 and the post router in each block. Randomize policy order within each block; one worker, four repetitions at each of 10 and 26 requests per second, 240 queries per trace. Measure service and queue-inclusive response time. Compare against the frozen loss margin and latency target. Since this control was added after seeing the original comparison, report it as a follow-up.
