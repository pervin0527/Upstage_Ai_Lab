import json

def main():
    file_path = "/home/pervinco/Upstage_Ai_Lab/Final/IR/src/outputs/UP-ER-QEN-CRV3.csv"
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.readlines()

    # Parse the JSON lines from the file
    parsed_data = [json.loads(line) for line in file_content]

    # Extract relevant fields and display them in a readable format
    def display_predictions(data):
        for entry in data:
            print(f"Eval ID: {entry['eval_id']}")
            print(f"Query: {entry['standalone_query']}")
            print(f"Answer: {entry['answer']}")
            print(f"TopK References (IDs): {', '.join(entry['topk'])}")
            print("\nReferences:")
            for ref in entry['references']:
                print(f"- Score: {ref['score']:.3f}")
                print(f"  Content: {ref['content']}\n")
            print("\n" + "="*50 + "\n")

    # Display the parsed predictions in a readable format
    display_predictions(parsed_data)

if __name__ == "__main__":
    main()