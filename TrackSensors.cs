using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using System.IO;

public class TrackSensors : MonoBehaviour
{
    private InputDevice headDevice;
    private List<string> data = new List<string>();

    void Start()
    {
        Debug.Log("Hello world!");
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.HeadMounted, devices);

        if (devices.Count > 0)
        {
            headDevice = devices[0];
            Debug.Log("Device found.");
        }
        else
        {
            Debug.Log("No device found!");
        }




    }

    void Update()
    {

        if (headDevice.isValid)
        {
            Vector3 acceleration;
            Vector3 angularVelocity;

            headDevice.TryGetFeatureValue(CommonUsages.deviceAcceleration, out acceleration);

            headDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out angularVelocity);

            Debug.Log("Acceleration: " + acceleration.ToString() + " " + "Angular Velocity: " + angularVelocity.ToString());
            string entry = $"{Time.time},{acceleration.x},{acceleration.y},{acceleration.z},{angularVelocity.x},{angularVelocity.y},{angularVelocity.z}";
            data.Add(entry);
        }
        else
        {
            Debug.LogError("HMD device is not valid");
        }
    }

    void OnApplicationQuit()
    {
        System.IO.File.WriteAllLines("sensor_data.csv", data.ToArray());
    }
}
