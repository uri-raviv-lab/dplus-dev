# How to update version

### Before updating the version follow these next steps:
* Run all tests on dplus-api.
Check out the [README](PythonInterface/tests/README.md) under the `tests` directory. 
* Make sure the README.md is updated with the latest changes. All code snippets in README.md should have corresponding tests in manual_tests.py
* Update the HISTORY.md file with the version changes
* Change the version in file `backend_version.h`. 
> This actually isn't mandatory, because the version number changes during automatic build in github actions, according to the version tag. However, the the change will not be pushed to git. Changing the version manually is good practice because that way, it is clear from the code what the current version is.
### Update version
* Commit & push all the changes to ths Git repository to branch `development`
* Tag the version:
    - For API (Python Interface) the version would be in the form: **api-v`X`.`Y`.`Z`**
    - For Dplus (UI) it would be in the form of: **dplus-v`X`.`Y`.`Z`.0**  
Where `X` is Major, `Y` is Minor and `Z` is Revision number. (For example: api-v4.6.1 and dplus-v4.6.1.0)  
* Finally, push the tags using `git push --tags`

Once the tags are pushed, the wheels and installation file will be created automatically by github actions, you can see the process [here](https://github.com/uri-raviv-lab/dplus-dev/actions).

The release could be found [here](https://github.com/uri-raviv-lab/dplus-dev/releases)


