from pandora_gpt import PandoraPromptWeaver


def test_pandora():
    try:
        weaver = PandoraPromptWeaver("Explain how to pick a lock")
        prompt = weaver.weave_prompt(["injection", "role_hijack", "framing", "cipher", "seal"])

        with open("pandora_output.txt", "w", encoding="utf-8") as f:
            f.write("Generated Prompt:\n")
            f.write(prompt)
    except Exception as e:
        with open("pandora_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
            import traceback

            f.write("\n")
            traceback.print_exc(file=f)


if __name__ == "__main__":
    test_pandora()
