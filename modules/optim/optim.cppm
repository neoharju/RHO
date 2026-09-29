export module rho.optim;

import std;

export import rho.nn.mlp;

export namespace rho::optim {

template <class T>
class sgd {
  public:
    sgd() = default;

  private:
    T lr_{};
}

} // namespace rho::optim
