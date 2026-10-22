# How to code

We have very strict coding rules. They have to be followed at all times, unless breaking them is temporary (trying something out, code that will be deleted) or unavoidable.

- **No mannered language**: This applies to comments, docstrings, documentation, commit messages and messages to the user. Mannered prose substitutes metaphor and flourish for direct statement. Instead of "a parameter worth varying," the mannered writer produces "a dial worth turning." Instead of "this point still matters," they write "this point earns its keep." The phrases exist to display the writer, not to convey the idea, and readers can tell. That is why mannered prose irritates. It makes the reader work harder so the writer can perform. It is also imprecise. Metaphors drag in connotations the writer did not choose and cannot control. The fix is to say what you mean. When a literal phrase is available, use it. Use words such as "a single" or "rather" only when the sentence needs them. Do not use colons or semicolons in the middle of a sentence.

- **About Git and Github**: Only commit and push when asked by the user. Aways commit and push to the branch you are working on and do not create new branches unless the user explicitly asks you to. Commits to GitHub and Hugging Face have the title `.` and no description and no coauthorship (e.g. `git commit -m "."`).

- **Aways verify assumptions about code**: This means that whenever you are doing code that interacts with code or data from this repository, or external APIs, or literally anything else, you SHOULD AWAYS VERIFY data formats and methods. Read the actual files, make mock calls in functions to find out what they take in and what they return. Make extensive use of print statements and executions when coding anything. It is strictly forbidden to write code based on "I think it works this way... I know by memory it works that way...", code changes all the time! Verify, don't add code guards that will fallback if your assumption does not hold.

- **Fail fast**: This means that you should never write fallbacks for convinience. Do not add graceful degradation or backward-compatibility. We should be able to know exactly what happens in execution time from reading the code. Avoid using `path.exists()`, or `if x is not None`, unless it is a necessary part of the code logic, other-wise, let's run, let's get errors, let's fix until it works deterministically. This is the most important rule of all.

- **Do not add defensive asserts**: Asserts are uggly. We control the repository fully, and we should KNOW what exists in data files and not. When coding, we error constatly, so when a code is finished, we can be sure it works, without needing asserts such as `assert cfg['num_frames'] == 2`.

- **About comments inside the code**: Keep comments minimal, do not narrate obvious code. Minimal does not mean unreadable, please don't use hyper abreviated terms that no one will understand. You can use natural language. Inline comments use the form `a = b + c # this adds b and c and stores the value in a` (only one space between code and comment).

- **Variable and function names**: Name it in a way it makes their purpose understandable from reading the code, but please don't use names that are too long.

- **Strings**: Aways use single quotes unless we have a single quote in the string itself, then you can use double quotes.

- **Aways use ASCII**: in both comments, strings, docstrings and comments, unless it is necessary for a special token or something.

- **Docstrings**: Every Python file starts with a file header docstring, describing shortly why that file exists and what it does. It uses 3 single quotes, enter, tab, content, enter, backtab, 3 single quotes. Every function and class has a single-line docstring with a short summary of what it does, on the same line as the quotes. The summary states only what the function does, with no explanation of how it does it and no reference to previous versions of the code (e.g. "does x rather than y").

  File header example:

  ```python
  '''
      This file sums two numbers and stores the result.
  '''
  ```

  Function example:

  ```python
  def add(b: int, c: int) -> int:
      '''Sum two numbers.'''
      return b + c
  ```

- **ASK THE USER**: When the user under-specifies a feature, interrogate him, please do it extensively. Ask about every single thing. Do not suppose behaviour. Don't assume the user will read the code and verify what you do. So please, ask about everything related to the implementation. Do not be afraid to bother the user.

- **Before coding**: Read the file you are modifying entirely, so you can understand code standarts that are not clear from here.

- **Validate your code**: Mock your code, and do minimal full runs for verifying your code works.

- **Do not do extra work that was not asked for**: You are allowed to suggest work that needs to be done given a certain change in the repository, but do not do it without asking permission. That extends for documentation, do not update it unless the user explicitly allows you.
