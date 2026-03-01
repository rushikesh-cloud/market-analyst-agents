
lets add agent level api calling in the current notebook
agents are right now using 'input' dict block, lets change this to {'messages': ['query']} based format required by langchain create_agent
at a api level validate the the input and output data sources in a accurate way

lets create a AGENTS.md file, include details of the whole code and also add python executable of uv environment python

lets adding logging and add comments in agents.md to add logging for each new modules

now lets add next js based frontend to connect to the individual agents apis and show case the results wiht source details, show the final results in markdown based viewer, show all the blocks with different left tabs
upload pdf document with ticker name to vector db
all the agents
extracting and showing web-search source links and answer text (not just raw JSON) so every agent block includes human-readable results with source details.

lets add gitignore file

in document ingestions lets add the ticker name details and add it to the metadata
lets create a endpoint to get all the ingested documents list in the dropdown for all the fundmental, technical and news based frontend uis, in document ingestion also show all the ingested document details with details and option to delete it from the vector db


now lets convert the agents to streaming agents so that we can showcase the live thinking and message details in frontend
add two options for each agent, one for direct invoke and one for streaming invoke
for streaming show each message type separately

now lets dump each message from the agent to database with time, company, input query, final messages, total tokens, number of messages from list

lets now add a dockerfile and nginx setup on top of it, both backend and frontend would be deployed together, backend would be available over /api endoint
and this would be deployed to azure container instances so setup using az command with already login

lets add a guardrails with smaller category model for faster response on all the agents, to validate the question is only related to markets around the company

now lets add a memory enabled session based chatbot to query the superviser agent, add a new tab in frontend for chat based option, it shoudl have a option to create a new session and showcase any sessions complete chat history, output should be enabled with markdown

use postgres for persistent memory, you can search for postgres checkpointer for langgraph install the library and configure and use it


Extra Things
- memory integrated
- Adding semantic caching to same cost and tokens usage
- Improving rag with reranking if required
- Use async to parallelly call the agent tools for speed up but causes extra memory and cpu
