import React from 'react';
import ReactDOM from 'react-dom/client';
import ChatApp from '../App'; // ✅ this is your chat UI

export function FloatingChat() {
  const containerId = 'floating-chat-container';

  if (document.getElementById(containerId)) return;

  // Create toggle button
  const toggleBtn = document.createElement('button');
  toggleBtn.innerText = '💬';
  Object.assign(toggleBtn.style, {
    position: 'fixed',
    top: '12px',
    right: '20px',
    zIndex: '9999',
    background: 'white',
    border: '1px solid #ccc',
    borderRadius: '50%',
    padding: '8px',
    fontSize: '20px',
    cursor: 'pointer',
  });

  // Create container for chatbox
  const chatDiv = document.createElement('div');
  chatDiv.id = containerId;
  Object.assign(chatDiv.style, {
    position: 'fixed',
    top: '60px',
    right: '20px',
    width: '400px',
    height: '600px',
    backgroundColor: 'white',
    border: '1px solid #ccc',
    boxShadow: '0px 4px 12px rgba(0,0,0,0.15)',
    zIndex: '9999',
    display: 'none',
  });

  toggleBtn.onclick = () => {
    chatDiv.style.display = chatDiv.style.display === 'none' ? 'block' : 'none';
  };

  document.body.appendChild(toggleBtn);
  document.body.appendChild(chatDiv);

  const root = ReactDOM.createRoot(chatDiv);
  root.render(<ChatApp />);
}
