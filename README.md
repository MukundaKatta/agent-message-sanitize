# agent-message-sanitize

Sanitize LLM message lists before sending. Zero dependencies.

```python
from agent_message_sanitize import sanitize_messages, clean_messages

result = sanitize_messages(messages)
# removes None content, normalizes roles, drops empty blocks
clean = result.messages
```

## Install

```bash
pip install agent-message-sanitize
```
