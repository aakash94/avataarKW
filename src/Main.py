from create_dataset import create_dataset
from nerf_main import do_nerf
from loadnshow import generate_video


def main():
    create_dataset()
    do_nerf()
    generate_video()


if __name__ == '__main__':
    print("Hello World")
    main()
