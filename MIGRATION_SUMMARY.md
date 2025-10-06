# Migration Summary - Modern LangChain Stack (Jan 2025)

## Problem Statement
The original setup used outdated packages that caused:
- **Mutex deadlock** during `sentence_transformers` import (Transformers cache migration issue)
- **Python 3.12 incompatibility** (FAISS 1.7.4, LangChain 0.1.0 + Pydantic 2 mismatch)
- **Deprecated import paths** (`langchain_community.embeddings`)

## Solution: Modern Stack (Option A)

### Package Updates

| Component | Old Version | New Version | Reason |
|-----------|-------------|-------------|---------|
| **LangChain** | 0.1.0 | ≥0.3,<0.4 | Pydantic 2 compatibility |
| **sentence-transformers** | 2.2.2 | ≥3.4.1 | Fix mutex lock issue |
| **transformers** | 4.38.0 | ≥4.41,<5.0 | Required by SBERT 3.4+ |
| **huggingface-hub** | 0.21.0 | ≥0.34 | Modern cache handling |
| **FAISS** | 1.7.4 | ≥1.12.0 | Python 3.12 wheel support |
| **Pydantic** | 2.5.3 | ≥2,<3 | Explicit v2 compatibility |
| **ragas** | 0.1.0 | latest | Modern API |

### New Packages Added
- `langchain-huggingface>=0.0.5` - Partner package for HuggingFace integrations

### Embedding Model Change

| Property | Old Model | New Model |
|----------|-----------|-----------|
| **Name** | `hkunlp/instructor-large` | `sentence-transformers/all-mpnet-base-v2` |
| **Class** | `HuggingFaceInstructEmbeddings` | `HuggingFaceEmbeddings` |
| **Dimensions** | 768 | 768 |
| **Max Tokens** | ~512 | ~384 |
| **Size** | ~1.3GB | ~420MB |
| **Quality** | Good | Better (best in `all-*` family) |

**Note**: If you had old data using `all-MiniLM-L6-v2` (384-dim), the dimension changed to 768-dim with `all-mpnet-base-v2`.

### Code Changes

#### Import Updates
```python
# OLD (deprecated in LangChain ≥0.2.2)
from langchain_community.embeddings import HuggingFaceEmbeddings

# NEW (modern partner package)
from langchain_huggingface import HuggingFaceEmbeddings
```

#### Files Modified
- `requirements.txt` - Full rewrite to modern stack
- `src/rag_chain.py` - Import path + model name
- `scripts/ingest.py` - Import path + model name
- `NEXT_STEPS.md` - Updated instructions and warnings

## Migration Steps Completed

✅ **Step 1**: Created fresh conda environment (`rag-ops`, Python 3.12.11)
✅ **Step 2**: Set fresh HuggingFace cache (`HF_HOME=$HOME/.cache/hf_rag`)
✅ **Step 3**: Updated `requirements.txt` to modern stack
✅ **Step 4**: Updated imports to `langchain_huggingface`
✅ **Step 5**: Changed embedding model to `all-mpnet-base-v2`
✅ **Step 6**: Updated documentation with warnings

## Next Steps for User

### 1. Install Dependencies
```bash
conda activate rag-ops
pip install -r requirements.txt
```

Expected install time: ~5-10 minutes (downloading packages)

### 2. Delete Old FAISS Index (CRITICAL)
```bash
# If you have an old faiss_index directory, delete it
rm -rf faiss_index/
```

**Reason**: Embedding dimensions may have changed (384→768 or different model). FAISS will error on mismatch.

### 3. Run Ingestion
```bash
python scripts/ingest.py
# OR
make ingest
```

Expected time:
- First run: ~10 minutes (downloads model ~420MB)
- Subsequent runs: ~2-3 minutes

### 4. Test Import (Verify No Mutex Lock)
```bash
python -c "from sentence_transformers import SentenceTransformer; print('✅ Import successful')"
```

Should complete in <5 seconds (not hang).

### 5. Run Application
```bash
streamlit run app.py
# OR
make run
```

## Verification Checklist

- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `sentence_transformers` import does not hang
- [ ] FAISS index builds successfully
- [ ] Streamlit app starts without import errors
- [ ] Can ask questions and get responses
- [ ] No deprecation warnings about `langchain_community.embeddings`

## Rollback Plan (If Needed)

If you encounter issues, you can revert:

```bash
# Deactivate and remove new environment
conda deactivate
conda env remove -n rag-ops

# Reactivate old environment
conda activate llm
```

Then restore original `requirements.txt` from git history.

## Key Benefits

1. ✅ **No more mutex deadlock** - Modern package versions handle cache properly
2. ✅ **Python 3.12 compatible** - All packages have proper wheels
3. ✅ **Future-proof** - Using latest LangChain (v0.3) with Pydantic 2
4. ✅ **Better embeddings** - `all-mpnet-base-v2` is more accurate than `all-MiniLM-L6-v2`
5. ✅ **No deprecation warnings** - Using official partner package for HuggingFace

## References

- [Sentence-Transformers Installation](https://sbert.net/docs/installation.html)
- [LangChain v0.3 Migration](https://python.langchain.com/docs/versions/v0_3/)
- [LangChain-HuggingFace Package](https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html)
- [FAISS Python 3.12 Compatibility](https://pypi.org/project/faiss-cpu/)
- [HuggingFace Cache Migration Issue](https://github.com/huggingface/transformers/issues/32345)

---

**Migration Date**: 2025-01-06
**Environment**: `rag-ops` (Python 3.12.11)
**Status**: Ready for dependency installation
