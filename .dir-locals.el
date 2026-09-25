;; Emacs settings for this project. The src/ modules are unity-build pieces:
;; flycheck checks each one alone, where functions used only by later modules
;; look unused. Same flags as .clangd. Indent with spaces, never tabs.
((c-mode . ((indent-tabs-mode . nil)
            (flycheck-clang-language-standard . "gnu17")
            (flycheck-clang-warnings . ("all" "no-unused-function"
                                        "no-pragma-once-outside-header"
                                        "no-undefined-internal")))))
