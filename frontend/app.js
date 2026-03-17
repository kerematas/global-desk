/*
  This file keeps the chat logic intentionally small.
  The goal is to make the first frontend easy to read, easy to demo,
  and easy to connect to a real backend later.
*/

const STORAGE_KEY = "global-desk-single-chat";

/*
  This starter message is shown on first load and after a reset.
  Keeping it in JavaScript means we can re-render it whenever needed.
*/
const STARTER_MESSAGES = [
  {
    role: "assistant",
    author: "Global Desk",
    text:
      "Hi! Ask a question about CPT, OPT, travel, taxes, work authorization, or maintaining status. This is a placeholder!"
  }
];

// Grab all of the elements we need once at startup.
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const chatMessages = document.getElementById("chatMessages");
const clearChatButton = document.getElementById("clearChatButton");
const sendButton = document.getElementById("sendButton");
const promptChips = document.querySelectorAll(".prompt-chip");

// Keep the current conversation in memory.
let messages = loadMessages();

// Track whether the interface is waiting on a response.
let isWaitingForReply = false;

/*
  Read saved messages from localStorage.
  If nothing is saved yet, fall back to the starter conversation.
*/
function loadMessages() {
  const saved = window.localStorage.getItem(STORAGE_KEY);

  if (!saved) {
    return [...STARTER_MESSAGES];
  }

  try {
    const parsed = JSON.parse(saved);
    return Array.isArray(parsed) && parsed.length > 0
      ? parsed
      : [...STARTER_MESSAGES];
  } catch (error) {
    console.error("Could not parse saved chat history:", error);
    return [...STARTER_MESSAGES];
  }
}

/*
  Save the current conversation so a refresh keeps the same single chat.
*/
function saveMessages() {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

/*
  Render the full message list from scratch.
  This is simple and perfectly fine for a small single-chat prototype.
*/
function renderMessages() {
  chatMessages.innerHTML = "";

  messages.forEach((message) => {
    chatMessages.appendChild(createMessageElement(message));
  });

  scrollToBottom();
}

/*
  Build one message row.
  The markup matches the CSS classes in styles.css.
*/
function createMessageElement(message) {
  const row = document.createElement("article");
  row.className = `message-row ${message.role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = message.role === "assistant" ? "GD" : "You";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const role = document.createElement("p");
  role.className = "message-role";
  role.textContent = message.author;

  const body = document.createElement("p");
  body.textContent = message.text;

  bubble.append(role, body);
  row.append(avatar, bubble);
  return row;
}

/*
  Add the temporary typing indicator while the assistant is "thinking."
*/
function showTypingIndicator() {
  const row = document.createElement("article");
  row.className = "message-row assistant";
  row.id = "typingIndicator";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "GD";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const role = document.createElement("p");
  role.className = "message-role";
  role.textContent = "Global Desk";

  const typing = document.createElement("div");
  typing.className = "typing-bubble";

  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("span");
    dot.className = "typing-dot";
    typing.appendChild(dot);
  }

  bubble.append(role, typing);
  row.append(avatar, bubble);
  chatMessages.appendChild(row);
  scrollToBottom();
}

/*
  Remove the typing indicator once we have a response.
*/
function hideTypingIndicator() {
  const typingIndicator = document.getElementById("typingIndicator");

  if (typingIndicator) {
    typingIndicator.remove();
  }
}

/*
  Small helper so the newest message stays visible.
*/
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/*
  Keep the textarea height comfortable as the user types.
*/
function autoResizeInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
}

/*
  For now this returns a placeholder.
  Later, replace the inside of this function with a real fetch() call to your
  backend endpoint and keep the rest of the frontend exactly the same.
*/
async function getAssistantReply(userMessage) {
  await delay(700);
  return buildPlaceholderReply(userMessage);
}

/*
  This placeholder text keeps the demo honest.
  It looks like a working chat, but it clearly says where the real answer will go.
*/
function buildPlaceholderReply(userMessage) {
  const loweredMessage = userMessage.toLowerCase();
  let topic = "your question";

  if (loweredMessage.includes("cpt")) {
    topic = "CPT";
  } else if (loweredMessage.includes("opt")) {
    topic = "OPT";
  } else if (loweredMessage.includes("tax")) {
    topic = "tax filing";
  } else if (loweredMessage.includes("travel")) {
    topic = "travel";
  } else if (loweredMessage.includes("status") || loweredMessage.includes("f-1")) {
    topic = "maintaining F-1 status";
  } else if (loweredMessage.includes("work") || loweredMessage.includes("job")) {
    topic = "employment";
  }

  return `You asked about ${topic}. This is a placeholder response for the new frontend.`;
}

/*
  Lightweight delay helper used by the placeholder reply.
*/
function delay(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

/*
  Main submit handler for typed messages.
*/
async function handleSubmit(event) {
  event.preventDefault();

  const text = messageInput.value.trim();

  if (!text || isWaitingForReply) {
    return;
  }

  isWaitingForReply = true;
  sendButton.disabled = true;

  messages.push({
    role: "user",
    author: "You",
    text
  });

  saveMessages();
  renderMessages();

  messageInput.value = "";
  autoResizeInput();
  showTypingIndicator();

  try {
    const reply = await getAssistantReply(text);

    hideTypingIndicator();

    messages.push({
      role: "assistant",
      author: "Global Desk",
      text: reply
    });

    saveMessages();
    renderMessages();
  } catch (error) {
    console.error("Could not create assistant reply:", error);
    hideTypingIndicator();

    messages.push({
      role: "assistant",
      author: "Global Desk",
      text:
        "Something went wrong while generating the placeholder reply. Check the console and try again."
    });

    saveMessages();
    renderMessages();
  } finally {
    isWaitingForReply = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
}

/*
  Clear the current single conversation and restore the starter message.
*/
function handleClearChat() {
  messages = [...STARTER_MESSAGES];
  saveMessages();
  renderMessages();
  messageInput.value = "";
  autoResizeInput();
  messageInput.focus();
}

/*
  Clicking a starter prompt sends that question immediately.
  This makes the layout feel more interactive without adding extra UI.
*/
function handlePromptChipClick(event) {
  const prompt = event.currentTarget.dataset.prompt;

  if (!prompt) {
    return;
  }

  messageInput.value = prompt;
  autoResizeInput();
  chatForm.requestSubmit();
}

/*
  Let Enter send the message, while Shift + Enter still adds a new line.
*/
function handleComposerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
}

// Initial render on page load.
renderMessages();
autoResizeInput();

// Wire up the main interactions.
chatForm.addEventListener("submit", handleSubmit);
messageInput.addEventListener("input", autoResizeInput);
messageInput.addEventListener("keydown", handleComposerKeydown);
clearChatButton.addEventListener("click", handleClearChat);
promptChips.forEach((chip) => chip.addEventListener("click", handlePromptChipClick));
