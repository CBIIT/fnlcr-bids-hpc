* [fnlcr-bids-hpc](https://cbiit.github.io/fnlcr-bids-hpc)  
  * [documentation](https://cbiit.github.io/fnlcr-bids-hpc/documentation)  
    * **How to create a central Git repository**  
    * [How to make your code CANDLE-compliant](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_make_your_code_candle_compliant)  
    * [How to run CANDLE on Biowulf](https://cbiit.github.io/fnlcr-bids-hpc/documentation/how_to_run_candle_on_biowulf)  
  * [How to use the image_segmentation module](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation)  
    * [packages](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages)  
      * [image_segmentation](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages/image_segmentation)  
    * [sample_data](https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/sample_data)  
  * [modules](https://cbiit.github.io/fnlcr-bids-hpc/modules)  
    * [candle](https://cbiit.github.io/fnlcr-bids-hpc/modules/candle)  
  * [native_candle_build_on_biowulf](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf)  
    * [build_on_2019-04-04](https://cbiit.github.io/fnlcr-bids-hpc/native_candle_build_on_biowulf/build_on_2019-04-04)  
  * [scripts](https://cbiit.github.io/fnlcr-bids-hpc/scripts)  


---

# How to create a central Git repository

## Use case #1

You want to create a new, empty repository in order to track your work.

### (1) Create a repository

1. Create a new BARE repository:

    ```bash
    cd /path/to/repo
    git init --bare myreponame.git
    ```

    **Note:** Add the `--shared` option to the `init` command if you eventually want the code to be shared with others in your group.

2. **(If you want others to be able to use the repository)** Ensure the repository permissions are correct:

   ```bash
   cd /path/to/repo
   chmod -R g+rw myreponame.git
   ```

### (2) Use the repository as normal

Clone the repository into a working directory:

```bash
cd /path/to/working/dir
git clone [myusername@biowulf.nih.gov:]/path/to/repo/myreponame
```

Use `[myusername@biowulf.nih.gov:]` in the `clone` command if the repository you just created is on a remote machine such as Biowulf.

Now changes can be pushed/pulled to/from the new repository.

## Use case #2

You have an unversioned tree of code and want to make a repository out of it in order to track future changes, whether by you only (Section 2a) or by both you and others (Section 2b).

### (1) Create a repository directly in the code tree

```bash
cd /path/to/code/tree
git init [--shared]
git add .
git commit -m "My commit message"
```

Add the `--shared` option to the `init` command if you eventually want the code to be shared with others in your group (Section 2b).

Note: Following ONLY this step will mean the code can't be cloned from elsewhere (it can only be committed to).

### (2a) Create a repository elsewhere in order for the code to be cloned/pushed/pulled **BY YOU ONLY**

1. Follow the steps in Section 1 (*without* the `--shared` option to `init`).

2. Clone that repository into a new BARE repository:

    ```bash
    cd /path/to/repo
    git clone --bare /path/to/code/tree myreponame.git
    ```

### (2b) Create a repository elsewhere in order for the code to be cloned/pushed/pulled **BY BOTH YOU AND OTHERS**

1. Follow the steps in Section 1 (*with* the `--shared` option to `init`).

2. Create a new BARE repository:

    ```bash
    cd /path/to/repo
    git init --bare --shared myreponame.git
    ```

3. Attach and push the code tree to the new repository:

    ```bash
    cd /path/to/code/tree
    git remote add origin /path/to/repo/myreponame
    git push --set-upstream origin master
    ```

4. Ensure the repository permissions are correct:

   ```bash
   cd /path/to/repo
   chmod -R g+rw myreponame.git
   ```

### (3) Use the repository as normal

Clone the repository into a working directory:

```bash
cd /path/to/working/dir
git clone [myusername@biowulf.nih.gov:]/path/to/repo/myreponame
```

Use `[myusername@biowulf.nih.gov:]` in the `clone` command if the repository you just created is on a remote machine such as Biowulf.

Now changes can be pushed/pulled to/from the new repository.
