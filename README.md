# BERT Playground

knp が死んだ！

```
brew install jumanpp
```

jumanpp コマンドが存在しないと死ぬようになっている(ホスト環境に依存するなマジで)

jumanpp コマンドのバージョン違いに気を付けろ！

手順

Release から最新の Juman を取ってくる

```
cd jumanpp-2.0.0-rc3/
mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make
```
