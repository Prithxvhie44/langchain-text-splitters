from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""Gotcha 👍 You just need some **dummy text with enough length + variation** to test a **length-based text splitter in LangChain**. Here you go — this is intentionally a bit long and uneven so chunking is obvious.

---

**Random Sample Text (for Text Splitter Testing)**

Artificial intelligence has rapidly evolved over the past decade, changing how humans interact with machines. What once required explicit instructions can now be achieved through learning patterns from massive datasets. This shift has enabled applications such as recommendation systems, natural language processing, computer vision, and autonomous systems.

In real-world scenarios, data is rarely clean or perfectly structured. Text may contain formatting inconsistencies, repeated ideas, or abrupt transitions between topics. For example, a technical document may suddenly switch from explaining a concept to showing an implementation detail or an error log. This makes effective text chunking extremely important when building retrieval-based systems.

LangChain provides multiple strategies for splitting text, including character-based, token-based, and recursive splitting. Choosing the correct strategy depends on the downstream task. A summarization pipeline may prefer larger chunks for context, while a question-answering system benefits from smaller, precise chunks.

Consider a system that processes user-uploaded PDFs. These files may include long paragraphs, short bullet points, and even empty lines. Without proper chunking, embeddings may lose semantic meaning or exceed model limits. Therefore, testing chunk size, overlap, and separator logic becomes critical during development.

As experimentation continues, developers often generate synthetic or random text to validate their pipelines. This helps ensure that edge cases such as uneven paragraph lengths, mid-sentence splits, and contextual overlap are handled correctly before deploying to production.

The ultimate goal is simple: preserve meaning while respecting model constraints.

---

If you want:

* **Much longer text (10k+ chars)**
* **Single paragraph only**
* **Highly repetitive text**
* **Technical / story / random words style**

Just tell me the format and I’ll generate it 🔧✨"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)
print(chunks)

