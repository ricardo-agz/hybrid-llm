import asyncio
import sys
from dynamic_model import DynamicModel, DynamicModelVariant
from utils import build_prompt

# ANSI escape codes for colors
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


async def chat(dynamic_model: DynamicModel):
    conversation = []
    print(f"** Tokens in {BLUE}blue{RESET} are generated by the small model. Tokens in {RED}red{RESET} are generated by the large model. **")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!")
                break

            # Append user message to the conversation
            conversation.append({"role": "user", "content": user_input})

            # Build the prompt from the conversation
            prompt = build_prompt(conversation)

            print("Assistant: ", end="", flush=True)
            response_content = ""

            # Generate and display the response token by token
            async for _ in dynamic_model.generate_response(prompt):
                # printing is handled in the async generator
                pass

            print()

            # Append assistant response to the conversation
            conversation.append({"role": "assistant", "content": response_content})

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            break


def main():
    dynamic_model = DynamicModel(variant=DynamicModelVariant.LOCAL)

    asyncio.run(chat(dynamic_model))


if __name__ == "__main__":
    main()
