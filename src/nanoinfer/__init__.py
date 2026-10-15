"""nanoinfer - a from-scratch LLM inference engine.

Built incrementally, each milestone answering one engineering question:

    M1  What are we actually bound by?          (roofline.py)
    M2  Why is a KV cache worth having?         (kv_cache.py)
    M3  Why does the KV cache waste memory?     (paged.py)
    M4  How should a server schedule requests?  (serving/)
    M5  How do we shrink what we read?          (quantization/)
    M6  How do we generate fewer tokens?        (speculative/)
"""

__version__ = "0.1.0"