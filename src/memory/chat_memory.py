"""Chat memory module for storing conversation history in the RAG system."""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class ChatMemory:
    """Stores and manages conversation history for the RAG system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config settings."""
        self.config = config
        self.history = {
            "messages": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }
        self.history_file = self._get_history_file_path()
        self._load_history()
    
    def _get_history_file_path(self) -> str:
        """Get the path to the history file."""
        # Use the memory directory from config or default to project root
        memory_dir = self.config.get("memory", {}).get("directory", "data/memory")
        
        # Create directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Use the filename from config or default to chat_history.json
        filename = self.config.get("memory", {}).get("filename", "chat_history.json")
        
        return os.path.join(memory_dir, filename)
    
    def _load_history(self) -> None:
        """Load conversation history from file if it exists."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"Loaded chat history with {len(self.history['messages'])} messages.")
            except Exception as e:
                print(f"Error loading chat history: {e}")
                # Initialize with empty history if loading fails
                self.history = {
                    "messages": [],
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                }
        else:
            print("No existing chat history found. Starting with empty history.")
    
    def _save_history(self) -> None:
        """Save conversation history to file."""
        try:
            # Update the last modified timestamp
            self.history["metadata"]["updated_at"] = datetime.now().isoformat()
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history."""
        message_entry = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        self.history["messages"].append(message_entry)
        self._save_history()
    
    def add_assistant_message(self, message: str, retrieved_documents: Optional[List[Dict]] = None) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: The assistant's response text
            retrieved_documents: Optional list of retrieved documents with metadata
        """
        message_entry = {
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add metadata about retrieved documents if provided
        if retrieved_documents:
            message_entry["metadata"] = {
                "retrieved_documents": retrieved_documents
            }
        
        self.history["messages"].append(message_entry)
        self._save_history()
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            max_messages: Optional maximum number of most recent messages to return
            
        Returns:
            List of message dictionaries
        """
        if max_messages:
            return self.history["messages"][-max_messages:]
        return self.history["messages"]
    
    def get_context_from_history(self, max_messages: int = 5) -> str:
        """
        Get a formatted string of recent conversation history for context.
        
        Args:
            max_messages: Maximum number of most recent messages to include
            
        Returns:
            Formatted string of conversation history
        """
        recent_messages = self.get_history(max_messages)
        context = ""
        
        for msg in recent_messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            context += f"{role}: {content}\n\n"
        
        return context.strip()
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = {
            "messages": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }
        self._save_history()
        print("Chat history cleared.")
    
    def get_relevant_history(self, query: str, max_messages: int = 5) -> List[Dict[str, Any]]:
        """
        Get messages from history that might be relevant to the current query.
        This simple implementation just returns the most recent messages.
        
        A more advanced implementation could use semantic similarity to find relevant messages.
        
        Args:
            query: The current user query
            max_messages: Maximum number of messages to return
            
        Returns:
            List of relevant message dictionaries
        """
        # For now, just return the most recent messages
        return self.get_history(max_messages)