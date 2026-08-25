## Ersatz Wildcards ##
### Modified from the Automatic1111 implemention: ###
#### https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards/ ####

---
* wildcard files go in `wildcards` directory in main webui directory.
    * plain text only;
    * loads from subdirectories too, directory character in wildcards is `/`: `__locations/fantasy__`;
    * one wildcard substitution per line (duplicates will be removed on load, a blank line is acceptable);
    * entries can reference other wildcard files. If you set up a loop, that's your problem;
    * lines starting with `#` are considered comments, not substitution options.
    * commands cannot be used in wildcard files, only directly in prompts - the files are lists of possible substitutions only
        * if you want to save a particular prompt, including commands, save it as a Style
        * if you must save your prompts as a wildcard, load with command `!$__wildcard__`, this will preload the substitution into the prompt before command processing. This works nested.
* use wildcards in prompts by referencing the wildcard file: `__{wildcard filename}__`.
    * a wildcard file called `colour.txt` would be referenced as `__colour__`;
    * adding `@` after the first delimiter, `__@colour__`, enables sequential substitution instead of random:
        * useful when generating a batch, especially when combined with command to lock seed;
        * index is reset to zero each generation.
    * adding '<' after the first delimiter, `__<colour__` reuses the previous result;
        * this is convenient for maintaining consistency without setting a variable, though more limited. It works inside wildcard substitutions, including when nested.
    * wildcards which do not match will be passed through as text, without the delimiters: `__jfgdejgjhdf__` *probably* will not match, therefore will become `jfgdejgjhdf`.
* *commands*:
    * `!${wildcard}` non-recursively apply the wildcard.
        * This happens first, for all such wildcards in the prompt.
    * `!{option1}:{option2}:{option3}:...` picks one of the options randomly;
        * options can be wildcards or variables
        * NOT WORK: !!a=__b__:!a=text_c (select between set variable a to wildcard b and set variable a to text_c)
        * WORK: !a=!b:c (set variable a to selection between b and c)
        * WORK: !a=b&c (see variables:append)
    * `!{count}*{text}` repeats {text}, {count} times with spaces in between.
        * if {text} is a wildcard, it may substitute to the same result multiple times;
        * does repeat variables, but that's probably not useful;
        * does work as an option in random pick;
        * potential use, picking a bunch of quality descriptors: `!7*__quality__`.
    * `!L` *must be at start of prompt* locks the seed used by all images in a batch to the seed of the first image.
    * `!ADD_START {prompt}` *(for HighRes prompts)* adds the {prompt} text to the start of the first pass prompt and uses the result for the HighRes pass;
        * must be at start of prompt (following `!L` if that is used);
        * This happens last, using the results of the first pass substitutions.
    * `!ADD_END {prompt}` *(for HighRes prompts)* does the end-of-prompt equivalent; use none or one only.
* *variables*:
    * `__{variable}__` **[GET]**
        * access variables like wildcards;
        * will be replaced in the prompt by the previously set value;
        * if the name doesn't match, it will be passed through as text: `__notarealvariable__` --> notarealvariable;
        * variables are used preferentially, so if one is named the same as a wildcard it will override that wildcard.
    * `!{variable}=text` **[SET]**
        * `!colourStorage=__colour__` [example result: incarnadine] *simple storage of one wildcard substitution*;
        * `!drkSouls=dark^__colour__` [example result: dark orange] *combined text and wildcard*;
        * `!bob=__shade__^__colour__` [example result: matte black] *two wildcards combined*;
        * Use `^` to chain together text(s) and wildcard(s), they will be replaced with ` ` (space);
        * values are resolved when the variable is set and other commands in the text are processed.
    * `!{variable}&text` **[APPEND]**
        * adds an option to a variable, so using the variable is a random pick between the options;
            * equivalent to setting multiple variables and using the pick command.
        * text can be a wildcard;
        * if variable doesn't exist, it will be created;
        * this can be combined with SET: `!var=__shade__&__shade__` makes {var} a choice between two randomly picked options from the shade wildcard or variable.
    * `!{variable}+text` **[EXTEND]**
        * extends all options of previously set variable;
        * text can be a wildcard.
        * if variable doesn't exist, it will be created;
        * this can be chained with SET: `!var=__shade__+__colour__`. In this case it is equivalent to '^'.
    * variables are set per processing run; a variable set in the first-pass prompt is available in other prompts too. Prompt processing order is: Prompt, Negative prompt, HighRes prompt, HighRes negative prompt.
* **Setting** to load wildcard files on webui load, or every run.

---
### More complex examples: ###
* `!c=__colour__ !k=dark+__c__ !l=light+__c__ person with !__k__:__l__:red hair and wearing __k__ __clothes_lower__ and __l__ __clothes_upper__ in __location_sf__`
    * sets variable `c` to a value (presumably a colour) from the wildcards file `colour.txt`;
    * sets variable `k` to `dark {c}` where {c} is the colour previously randomly chosen;
    * sets variable `l` to `light {c}` - we now have dark and light variants of the same random colour;
    * randomly selects either `k` or `l` or `red` for hair colour;
    * uses both dark/light variants for a type of clothing;
    * standard wildcard usage for a location;
    * example output: person with dark green hair and wearing dark green leggings and light green sweatshirt in Glass dome city beneath toxic crimson skies
* `!d=!3*ho !d&__colour__ !2*xmas+__d__`
    * sets variable `d` to `ho ho ho`;
    * adds colour wildcard to variable d;
    * adds `xmas ho ho ho {colour} xmas ho ho ho {colour}` to prompt.
* `!c=__colour__ !k=dark+__c__ !l=light+__c__ !c=__colour__ !k=dark+__c__ !l=light+__c__  !e=!__k__:__l__ __e__`
    * sets variable `c` to a value (presumably a colour) from the wildcards file `colour.txt`;
    * sets variable 'k' to `dark {c}` where {c} is the colour previously randomly chosen;
    * sets variable 'l' to `light {c}` - we now have dark and light variants of the same random colour;
    * sets variable `e` to a random selection of `k` or `l`;
    * adds `{e}` to the prompt: dark/light {colour}
    * So, it is possible to set up a random but consistent 'palette'.
* `!L __body__ woman with __face__ in __location_fantasy__ __@colour__`, when used with Batch, will generate a set of images with fixed seed and identical prompts - with the exception of the final wildcard, which will sequentially (and alphabetically, due to sorting) work through options in the colour wildcard file.