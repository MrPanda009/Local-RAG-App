# Local RAG App (Android)

An on-device LLM engine for Android with local vector and graph storage for retrieval. The goal is a fully offline, private RAG app: run a GGUF model through [llama.cpp](https://github.com/ggml-org/llama.cpp), and retrieve context from a local vector index and knowledge graph without any network calls.

> **Status: early-stage / work in progress.** The inference layer and the storage layer are written, but they are not yet wired together and the UI is still the Android Studio template. See [Status](#status) for exactly what works and what doesn't.

## Architecture

```
┌──────────────────────────────────────────────┐
│  Compose UI                (placeholder)      │
├──────────────────────────────────────────────┤
│  Kotlin engine                                │
│    LlamaEngine      model lifecycle, single-  │
│                     thread dispatcher         │
│    InferenceSession generate() → Flow<String> │
├──────────────────────────────────────────────┤
│  JNI bridge (C++17)        llama-bridge.cpp   │
├──────────────────────────────────────────────┤
│  llama.cpp (git submodule, CPU + ARM NEON)    │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  Retrieval storage                            │
│    VectorStore  ObjectBox + HNSW index        │
│    GraphStore   Room/SQLite, recursive CTE    │
└──────────────────────────────────────────────┘
```

### Inference engine

- **Native bridge** (`app/src/main/cpp/llama-bridge.cpp`): loads GGUF weights with `mmap`, creates a context and sampler per session, and streams generated token pieces back to Kotlin through a callback. Returning `false` from the callback cancels generation.
- **System prompt pre-fill:** the system prompt is decoded into the KV cache once when a session is created, and the position is tracked across turns, so each query only decodes its own new tokens.
- **Sampling:** temperature 0.8, top-k 40, top-p 0.9, min-p 0.05.
- **Safety checks:** a context-full guard before decoding, and end-of-generation detection.
- **Build:** CMake 3.22.1 with the NEON backend enabled. CPU-only for now; a Vulkan backend flag is stubbed out in `CMakeLists.txt`.
- **Kotlin side:** `LlamaEngine` is a singleton that owns the model pointer and serializes all native calls on one dedicated thread. `InferenceSession.generate()` exposes the token stream as a coroutines `Flow<String>`.

### Retrieval storage

- **Vector store:** `Chunk` entities in [ObjectBox](https://objectbox.io/) with a 384-dimension cosine HNSW index, queried with `findNearest()`.
- **Graph store:** `Entity` and `Relationship` tables in Room with the bundled SQLite driver. `getNeighborsWithinDepth()` walks relationships with a recursive CTE (default depth 2).

## Status

| Area | State |
|---|---|
| llama.cpp build via CMake / NDK | Written |
| JNI bridge: load model, create session, stream tokens, free | Written |
| Kotlin engine (`LlamaEngine`, `InferenceSession`) | Written |
| Vector store (ObjectBox + HNSW) | Written |
| Graph store (Room + recursive neighbor query) | Written |
| Kotlin ↔ native JNI binding | **Not working yet** (see below) |
| Embedding model / document ingestion | Not started |
| Entity and relationship extraction | Not started |
| Retrieval → prompt → generation pipeline | Not started |
| UI | Placeholder |
| Tests | Template only |

### Known issues

- **JNI signatures don't match the Kotlin declarations yet.** The native functions are exported under the package `dev.yourname.graphrag.engine`, but `LlamaEngine` lives in `com.example.local_rag_app.engine`, so the library will not bind at runtime. The parameter lists and return types for `nativeCreateSession`, `nativeGenerate`, and `nativeContextUsage` also differ between the C++ and Kotlin sides and need reconciling.
- The storage layer still uses the placeholder package name `dev.yourname.graphrag`.

## Building

Requirements: Android Studio, the Android NDK, CMake 3.22.1, and an arm64 device (the NEON backend is enabled).

```bash
git clone --recurse-submodules https://github.com/MrPanda009/Local-RAG-App.git
```

Open the project in Android Studio, let Gradle sync, and build. The llama.cpp submodule must be present, since `CMakeLists.txt` pulls it in with `add_subdirectory`.

You will need your own GGUF model file. None is bundled, and the app does not yet have a UI to load one.

## Roadmap

1. Reconcile the JNI package name and signatures, and get a first token stream running end to end on a device.
2. Add an on-device embedding model (the vector index expects 384-dimension embeddings).
3. Build document ingestion: chunking, embedding, and writing to `VectorStore`.
4. Extract entities and relationships into `GraphStore`.
5. Combine vector hits and graph neighbors into the prompt, then stream the answer to a Compose chat UI.

## Tech stack

Kotlin, Jetpack Compose, C++17, JNI, llama.cpp (GGUF), ObjectBox, Room, SQLite, Koin. Min SDK 29, target SDK 36.

## License

MIT. See [LICENSE](LICENSE).
