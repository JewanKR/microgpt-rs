# A Rust port of Andrej Karpathy's microgpt.py

A pure Rust port of Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). While preserving the original mathematical structure and educational clarity, this implementation aims for enhanced performance by leveraging Rust's static typing and efficient memory management.

## 🚀 Key Features & Optimizations

* **Memory Efficiency**: Minimized heap allocations with a flat parameter structure and smart pointers (`Rc`), ensuring efficient memory sharing across the computational graph.
* **Static Type Safety**: Replaced Python’s dynamic overhead with Rust’s compile-time optimizations.
* **Strongly Typed Architecture**: Replaced Python's dictionary-based parameter management with explicit structs, providing compile-time structural safety.
* **Bidirectional Arithmetic Support**: support bidirectional operations (e.g., `Value + f64` and `f64 + Value`). This ensures the mathematical implementation remains as clean and readable as the original Python code.

## 🌱 Porter's Note (A Learning Journey)
As a Rust beginner, my goal was to faithfully preserve the algorithmic clarity of the original microgpt.py while embracing the performance and idioms of Rust. I aimed to write code that feels like Rust but reads like the original Python.

Throughout the source code, you'll find "Porter's Notes" highlighting where and why I translated dynamic Python patterns into static Rust idioms. I hope these annotations help fellow developers bridging the gap between the two languages.

---
*Contributions, discussions, and feedback are always welcome!*
