# VECMA-hackathon

This repo contains examples for working with uncertainty
quantification using the VECMA toolkit.

# Cloning
Git submodules are used to bring in external dependencies. Please clone with either

~~~
git clone --recursive git@github.com:boutproject/VECMA-hackathon.git
~~~

or run

~~~
git submodules --init --recursive
~~~

following the initial clone to ensure these submodules are available. 

Note these submodules use the git protocol so do not allow anonymous access.

# Building

To build exectute

~~~
cmake . -B build && cmake --build build
~~~
