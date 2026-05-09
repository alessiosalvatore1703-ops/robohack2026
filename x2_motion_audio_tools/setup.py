from setuptools import find_packages, setup

package_name = "x2_motion_audio_tools"

setup(
    name=package_name,
    version="0.1.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/x2_person_follow.launch.py",
                "launch/x2_person_follow_debug.rviz",
                "launch/x2_person_track_torso.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robohack2026",
    maintainer_email="dev@robohack2026.local",
    description="AgiBot X2 voice, microphone, and motion helper nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "x2_bedrock_voice_assistant = x2_motion_audio_tools.x2_bedrock_voice_assistant:main",
            "x2_sound_logger = x2_motion_audio_tools.x2_sound_logger:main",
            "x2_mic_logger = x2_motion_audio_tools.x2_mic_logger:main",
            "x2_go_to_offset_raise_arms = x2_motion_audio_tools.x2_go_to_offset_raise_arms:main",
            "x2_turn_to_person_tts = x2_motion_audio_tools.x2_turn_to_person_tts:main",
            "x2_person_follow = x2_motion_audio_tools.x2_person_follow:main",
            "x2_person_track_torso = x2_motion_audio_tools.x2_person_track_torso:main",
            "x2_forward_back_raise_arms = x2_motion_audio_tools.x2_forward_back_raise_arms:main",
            "x2_forward_backward_steps = x2_motion_audio_tools.x2_forward_backward_steps:main",
        ],
    },
)
