                ┌───────────────────────────────┐
                │        Streamlit App UI       │
                │                               │
                │  ┌───────────────┐            │
                │  │   Title &     │            │
                │  │  Caption      │            │
                │  └───────────────┘            │
                │                               │
                │  ┌───────────────┐            │
                │  │  Chat Box     │ <--- User types coding questions
                │  │ (chat_input)  │
                │  └───────────────┘
                │                               │
                │  ┌───────────────┐            │
                │  │ Chat Display  │ <--- Shows messages from
                │  │ (message_log) │      AI & User
                │  └───────────────┘
                │                               │
                │  ┌───────────────┐            │
                │  │ Sidebar       │            │
                │  │ ┌───────────┐ │            │
                │  │ │ Model     │ │ <--- User selects model
                │  │ │ Dropdown  │ │
                │  │ └───────────┘ │
                │  │ ┌───────────┐ │
                │  │ │ Model     │ │ <--- Description of capabilities
                │  │ │ Features  │ │
                │  │ └───────────┘ │
                │  └───────────────┘
                └───────────────────────────────┘
                             │
                             ▼
                ┌───────────────────────────────┐
                │      Backend AI Flow          │
                │                               │
                │  1. Append user message       │
                │     to session_state          │
                │                               │
                │  2. Build Prompt Chain       │
                │     - System prompt           │
                │     - Previous messages       │
                │                               │
                │  3. ChatOllama LLM           │
                │     - Generate AI response    │
                │                               │
                │  4. StrOutputParser           │
                │     - Clean AI response       │
                │                               │
                │  5. Append AI response       │
                │     to session_state          │
                │                               │
                │  6. Streamlit refresh        │
                │     - Update chat display     │
                └───────────────────────────────┘
                             │
                             ▼
                     User sees updated chat
