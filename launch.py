from modules import launch_utils
list_extensions = launch_utils.list_extensions
args = launch_utils.args
run_pip = launch_utils.run_pip
is_installed = launch_utils.is_installed
run_extension_installer = launch_utils.run_extension_installer


def main():
    if launch_utils.args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not launch_utils.args.skip_prepare_environment:
            launch_utils.prepare_environment()

    if launch_utils.args.forge_ref_a1111_home:
        launch_utils.configure_forge_reference_checkout(launch_utils.args.forge_ref_a1111_home)

    launch_utils.start()


if __name__ == "__main__":
    main()
