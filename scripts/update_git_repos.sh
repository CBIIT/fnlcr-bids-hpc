#!/bin/bash

# Determine the checkouts directory containing the git repositories
checkouts_dir=${1:-$HOME/checkouts}

# Check the output of "git status" for a particular string
function check_git_status() {

    # Set parameter
    string_to_detect=$1

    # Search for the desired string outputted by the "git status" command
    git status | grep "$string_to_detect" > /dev/null

    # If the string was found...
    if [ $? -eq 0 ]; then
        echo 1

    # If the string was NOT found...
    else
        echo 0
    fi
}

# Ask to do something and optionally do it
function ask_to_do() {

    # Set parameters
    question=$1 # must be yes-or-no question
    positive_response=$2
    negative_response=$3

    # Output the "git status"
    git status

    # Ask the question and read the response
    echo -en "\n$question (y/Y/[n/N]) "
    read response

    # If the response is yes...
    if [ "a$response" == "ay" -o "a$response" == "aY" ]; then
        eval $positive_response

    # If the response is no...
    else

        # If the negative response is not blank...
        if [ ! "a$negative_response" == "a" ]; then
            eval $negative_response
        fi
    fi
}

# For each repository...
for dir in $checkouts_dir/*; do

    # Output the current repository
    echo -e "\n**** $dir ****\n"

    # Enter the current repository
    pushd $dir > /dev/null

    # Pull any changes from the remote repository
    git pull

    # Get the current repository name
    repo="repository $(basename $dir)"

    # Optionally add untracked files
    if [ $(check_git_status "^Untracked files:$") -eq 1 ]; then
        ask_to_do "Add untracked files to $repo (otherwise, update .gitignore)?" \
            "echo -e \"a\n*\nq\n\" | git add -i" \
            "echo -en \"Enter string to add to .gitignore: \"; read str; echo \$str >> .gitignore; git add .gitignore"
    fi

    # Optionally stage files for commit
    if [ $(check_git_status "^Changes not staged for commit:$") -eq 1 ]; then
        ask_to_do "Stage files for commit to $repo?" \
            "git add ." \
            ""
    fi

    # Optionally commit changes to the local repository
    if [ $(check_git_status "^Changes to be committed:$") -eq 1 ]; then
        ask_to_do "Commit changes to $repo?" \
            "git commit" \
            ""
    fi

    # Optionally push commits to the remote repository
    if [ $(check_git_status "^Your branch is ahead of ") -eq 1 ]; then
        ask_to_do "Push changes to $repo?" \
            "git push" \
            ""
    fi

    # Go back to the original directory from which this script was called
    popd > /dev/null

done