export module foo;

import std;

export namespace foo {

inline void hello_world() {
    std::println("Hello world from foo-module!");
}

} // namespace foo
