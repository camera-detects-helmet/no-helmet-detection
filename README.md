# A real-time detection rider not wearing a helmet useing DeepLearning (YOLOv5)


## Installation

- pytorch
- python 3.8 
- docker

install environment

``` bash
python3 -m venv venv
```

``` bash
source venv/bin/activate
```

``` bash
pip install -r requirements.txt
```


## Streaming installation
intallation RTSP server with docker (Server RTSP)
```bash
docker run --rm -it -e RTSP_PROTOCOLS=tcp -p 8554:8554 -p 1935:1935 -p 8888:8888 -p 8889:8889 aler9/rtsp-simple-server
```
start producer ( RPI )
``` bash
python3 rtsp_stream_producer.py   
```

start detection 
``` bash
python3 main.py   
```

start customer ( test camera )
``` bash
python3 rtsp_customer.py   
```


## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```


