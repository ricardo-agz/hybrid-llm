"use client";

import React, {
  useState,
  useEffect,
  useRef,
  FormEvent,
} from 'react';
import { Button, Input, Textarea, Switch, Modal } from "@/components/ui";
import { cn } from "@/lib/utils";
import { AiOutlineInfoCircle, AiOutlineRobot } from "react-icons/ai";


// Define the possible roles
type Role = 'system' | 'user' | 'assistant';

interface SystemMessage {
  role: 'system';
  content: string;
}

interface UserMessage {
  role: 'user';
  content: string;
}

interface AssistantContentItem {
  token: string;
  modelUsed: string;
}

interface AssistantMessage {
  role: 'assistant';
  content: AssistantContentItem[];
}

type Message = SystemMessage | UserMessage | AssistantMessage;

interface AssistantResponse {
  choices: Array<{
    delta: {
      content?: string;
    };
    metadata: {
      model_used: string;
    };
  }>;
}

interface APIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

const defaultSystemPrompt = 'You are an intelligent AI assistant trained to do complex reasoning. You should carefully think about problems before outputting your final response.';


function App() {
  const [conversation, setConversation] = useState<Message[]>([
    { role: 'system', content: defaultSystemPrompt }
  ]);
  const [userInput, setUserInput] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isLocalOnly, setIsLocalOnly] = useState<boolean>(true);
  const [maxTokens, setMaxTokens] = useState<number>(100);
  const [systemPrompt, setSystemPrompt] = useState<string>(defaultSystemPrompt);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const clearConversation = () => {
    setConversation([{ role: 'system', content: systemPrompt }]);
  }

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    const newConversation: Message[] = [
      { role: 'system', content: systemPrompt },
      ...conversation.slice(1),
      { role: 'user', content: userInput },
      { role: 'assistant', content: [] },
    ];
    setConversation(newConversation);
    setUserInput('');
    streamAssistantResponse(newConversation);
  };

  const streamAssistantResponse = (currentConversation: Message[]) => {
    setIsStreaming(true);

    const processedConversation: ({ role: string; content: string } | SystemMessage | UserMessage | AssistantMessage)[] = currentConversation.map((msg) => {
      if (msg.role === 'assistant' && Array.isArray(msg.content)) {
        return { role: 'assistant', content: msg.content.map((item) => item.token).join('') };
      }
      return msg;
    });

    const endpoint = isLocalOnly
        ? 'http://localhost:8080/chat'
        : 'http://localhost:8080/chat-local-cloud';

    fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation: processedConversation,
        max_tokens: maxTokens,
      }),
    })
        .then((response) => {
          if (!response.body) {
            throw new Error('ReadableStream not supported in this browser.');
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          const readChunk = () => {
            reader.read().then(({ done, value }) => {
              if (done) {
                setIsStreaming(false);
                return;
              }

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.trim() === '') continue;
                if (line.startsWith('data: ')) {
                  const dataStr = line.slice(6).trim();
                  if (dataStr === '[DONE]') {
                    reader.cancel();
                    setIsStreaming(false);
                    return;
                  } else {
                    try {
                      const data: AssistantResponse = JSON.parse(dataStr);
                      const token = data.choices[0].delta.content || '';
                      const modelUsed = data.choices[0].metadata.model_used;

                      setConversation((prev) => {
                        const newConversation = [...prev];
                        const lastMessageIndex = newConversation.length - 1;
                        const lastMessage = newConversation[lastMessageIndex];

                        if (lastMessage.role === 'assistant' && Array.isArray(lastMessage.content)) {
                          const updatedContent: AssistantContentItem[] = [
                            ...lastMessage.content,
                            { token, modelUsed },
                          ];
                          newConversation[lastMessageIndex] = {
                            ...lastMessage,
                            content: updatedContent,
                          };
                        }

                        return newConversation;
                      });
                    } catch (err) {
                      console.error('Failed to parse JSON:', err);
                    }
                  }
                }
              }
              readChunk();
            });
          };

          readChunk();
        })
        .catch((error) => {
          console.error('Fetch failed:', error);
          setIsStreaming(false);
        });
  };

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  const renderMessage = (message: Message, index: number) => {
    const isUser = message.role === 'user';
    const isAssistant = message.role === 'assistant';
    const isSystem = message.role === 'system';

    return (
        <div
            key={index}
            className={cn(
                'flex items-start',
                isUser ? 'justify-end' : isAssistant ? 'justify-start' : 'justify-center'
            )}
        >
          {!isUser && !isSystem && (
              <div className="flex-shrink-0 mr-2 bg-black rounded-md p-2">
                <AiOutlineRobot className="text-xl text-white" />
              </div>
          )}
          <div
              className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-2 rounded-lg shadow ${
                  isUser
                      ? 'bg-blue-500 text-white'
                      : isAssistant
                          ? 'bg-gray-200 text-gray-800'
                          : 'bg-yellow-100 text-gray-700 italic'
              }`}
          >
            {isAssistant && Array.isArray(message.content) ? (
                message.content.map((item, idx) => (
                    <span
                        key={idx}
                        className={item.modelUsed === 'small' ? 'text-blue-500' : 'text-red-500'}
                    >
                {item.token}
              </span>
                ))
            ) : (
                <p>{message.content.toString()}</p>
            )}
          </div>
        </div>
    );
  };

  return (
      <div className="flex flex-col h-screen bg-gray-50">
        {/* Header */}
        <div className="flex justify-between items-center p-4 bg-white shadow-md">
          <div className="flex items-center space-x-4 w-32">
            <Switch
                checked={!isLocalOnly}
                onCheckedChange={() => setIsLocalOnly(!isLocalOnly)}
                className="focus:ring-blue-500"
            />
            <span className="text-gray-600">{isLocalOnly ? 'Local' : 'Edge/API'}</span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="flex items-center text-sm text-blue-500">
              <AiOutlineRobot className="mr-2" /> Llama 3.2 1B
            </span>
            <span className="flex items-center text-sm text-red-500">
              <AiOutlineRobot className="mr-2" /> Llama 3.2 3B {isLocalOnly ? '' : `(API)`}
            </span>
            {/* Info Icon */}
            <button
                onClick={() => setIsModalOpen(true)}
                className="text-gray-600 hover:text-gray-800 focus:outline-none"
                aria-label="Open Information Modal"
            >
              <AiOutlineInfoCircle size={20} />
            </button>
          </div>
          <Button onClick={clearConversation} className="text-sm text-white">
            Clear
          </Button>
        </div>

        {/* System Prompt */}
        <div className="p-4 bg-white shadow-inner">
          <Textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Enter system prompt..."
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
          />
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 bg-gray-100">
          {conversation.slice(1).map((msg, idx) => renderMessage(msg, idx))}
          <div ref={messageEndRef} />
        </div>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="p-4 bg-white shadow-md flex items-center space-x-4">
          <Input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              disabled={isStreaming}
              placeholder="Type your message..."
              className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Button
              type="submit"
              disabled={isStreaming}
              className="flex items-center justify-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {isStreaming ? 'Sending...' : 'Send'}
          </Button>
        </form>

        <Modal
            isOpen={isModalOpen}
            onClose={() => setIsModalOpen(false)}
            title="Mixture of Depths Across Edge/API Models"
        >
          <p>
            <strong>What is this?</strong> This is a simple chat interface to interact with our beta version of a distributed model architecture.
          </p>
          <p className="mt-2">
            <strong>What is a distributed model?</strong> Notice how some of the tokens generated are red and some are blue, this is because different models were used to generate these tokens. A small model is used as the default until a point of low confidence is reached, at which point a larger model will be swapped in to continue generating tokens until a point of high confidence is reached and the small model can be swapped back in. This process is repeated throughout the length of the query.
          </p>
          <p className="mt-2">
            <strong>Why is this cool?</strong> The app allows you to have interactive conversations with an AI assistant. It processes your inputs, generates thoughtful responses, and can handle various topics and queries.
          </p>
        </Modal>
      </div>
  );
}

export default App;
