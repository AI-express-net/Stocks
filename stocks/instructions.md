- Read all the files called requirements.md.
- There can be multiple requirement files, check all the modules in the project.
- All code must follow the general coding rules in coding_rules.md
- After any changes are made to the code, run pytest on the code.
- Never suggest to use a mock in production code.
- All requirements in the requirements-file should be reflected in pytest unit-tests.
- All applications and pytest unit-tests need to run in PyCharm as wel as Cursor. Including making sure that any new packages introduced in Cursor are alo available in PyCharm.

The following instructions were suggested by Cursor itself:
- Mandating systematic search before making interface changes.
- Requiring all changes to be made at once rather than incrementally.
- Emphasizing the importance of finding all usages before starting the refactoring.

When I ask you to 'verify plan' then check if the text above still matches the code. When changes need to be made to update the code to match the above text, show me the changes you're planning. All new code for this back tester needs to go in the 'back_tester' folder. Please write the plan verification results in a file that has the current time-stamp in the name so I can keep a list of changes over time. Also write the whole plan, with all its phases in a file called 'back-tester-plan-' with the current timestamp appended to the filename.