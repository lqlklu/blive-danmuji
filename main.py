import argparse
import asyncio

import sounddevice
from blive import BLiver, Events, BLiverCtx
from blive.msg import DanMuMsg

from inference import load_model, inference


async def play(audio, sampling_rate):
    time = len(audio)/sampling_rate
    sounddevice.play(audio, samplerate=sampling_rate)
    await asyncio.sleep(time)


async def reading(queue, sampling_rate):
    while True:
        audio = await queue.get()
        if audio is None:
            break
        await play(audio, sampling_rate)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', type=str)
    parser.add_argument('-c', '--config-path', type=str)
    parser.add_argument('-id', '--room-id', type=int)
    parser.add_argument('-s', '--spk', type=str)
    parser.add_argument('-l', '--language', type=str, default='简体中文')
    parser.add_argument('-ns', '--noise-scale', type=float, default=0.667)
    parser.add_argument('-nsw', '--noise-scale-w', type=float, default=0.6)
    parser.add_argument('-ls', '--length-scale',
                        type=float, default=1, help='整体语速')
    args = parser.parse_args()
    model_path = args.model_path
    config_path = args.config_path
    room_id = args.room_id
    spk = args.spk
    language = args.language
    noise_scale = args.noise_scale
    noise_scale_w = args.noise_scale_w
    length = args.length_scale
    print(model_path, config_path, room_id, spk,
          language, noise_scale, noise_scale_w, length)

    net_g, hps = load_model(model_path=model_path,
                            config_path=config_path)

    queue = asyncio.Queue()
    app = BLiver(room_id)

    @app.on(Events.DANMU_MSG)
    async def on_danmu(ctx: BLiverCtx):
        danmu = DanMuMsg(ctx.body)
        uname = danmu.sender.name
        content = danmu.content
        print(uname, content)
        text = f"{uname} 说 {content}"
        audio = inference(net_g=net_g, hps=hps,
                          language=language, text=text, spk=spk, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length=length)
        await queue.put(audio)

    t1 = app.run_as_task()

    await asyncio.gather(t1, reading(queue, hps.data.sampling_rate))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
