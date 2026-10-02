# MNIST dataset needs to be mounted, they are not included in the image,
# so use the fetch script first to download the datasets.
#
# Build and run the program:
#   docker build -t rho .
#   docker run --rm rho
#   #docker run --rm -v "$PWD/data:/app/data:ro" rho
#
# Build and run the tests
#   docker build --target tests -t rho-tests .
#   docker run --rm -t rho-tests
#   #docker run --rm -t -v "$PWD/data:/src/data:ro" rho-tests

FROM archlinux:latest@sha256:b21322c663be387c0ed9cbc7bbbfe18e41633ad4e7b7c77cfad45f128be20040 AS base

ARG ARCH_SNAPSHOT=2026/10/01

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    echo "Server = https://archive.archlinux.org/repos/${ARCH_SNAPSHOT}/\$repo/os/\$arch" \
        > /etc/pacman.d/mirrorlist \
	&& pacman -Sy --noconfirm --needed archlinux-keyring \
	&& pacman -Suu --noconfirm

# build ---------------------------------------------------------------------
FROM base AS build

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -S --noconfirm --needed gcc cmake ninja git

WORKDIR /src
COPY CMakeLists.txt ./
COPY modules/ modules/
COPY src/ src/
COPY tests/ tests/

# We are making a static binary for runtime
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRHO_BUILD_TESTS=ON \
          -DCMAKE_EXE_LINKER_FLAGS=-static-pie \
 && cmake --build build

# tests ---------------------------------------------------------------------
FROM base AS tests
RUN useradd --uid 10001 --no-create-home --shell /usr/bin/nologin rho
COPY --from=build /src/build/RHO_tests /usr/local/bin/rho_tests

WORKDIR /src
USER 10001
ENTRYPOINT ["/usr/local/bin/rho_tests"]

# runtime -------------------------------------------------------------------
FROM scratch AS runtime

COPY --from=build /src/build/RHO /usr/local/bin/rho

USER 10001:10001
WORKDIR /app

ENTRYPOINT ["/usr/local/bin/rho"]
