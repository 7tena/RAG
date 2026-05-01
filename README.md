### Retrieval Augmented Generation
This repository covers the basics of RAG. It is split into 5 modules:
- Module 1: RAG Overview
  - Build a RAG system, by writing functions to augment the prompt with relevant retrieved documents and generating system and user dicts to pass into an LLM.
- Module 2: Information Retrieval and Search Foundation
  - Implement a function to perform semantic search using pre-trained embeddings and cosine similarity as part of the RAG framework, as well as BM25 retrieve and Reciprocal Rank Fusion. Explore how different retrieve functions impact the LLM output.
- Module 3:Information Retrieval with Vector Databases
  - Work with Weaviate API to scale up our RAG system, now working with a larger dataset of news items from the BBC. We will chunk this data, load it into a vector database, and perform a variety of retrieval techniques.
- Module 4:LLMs and Text Generation
  - Develop a RAG based chatbot.
- Module 5:RAG Systems in Production
  - Improving the chatbot.
