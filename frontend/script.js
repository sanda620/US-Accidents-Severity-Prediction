const form =
    document.getElementById(
        "predictionForm"
    );

const result =
    document.getElementById(
        "result"
    );

const errorBox =
    document.getElementById(
        "error"
    );

const severityResult =
    document.getElementById(
        "severityResult"
    );

const descriptionResult =
    document.getElementById(
        "descriptionResult"
    );

const probabilities =
    document.getElementById(
        "probabilities"
    );


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        result.classList.add(
            "hidden"
        );

        errorBox.classList.add(
            "hidden"
        );


        const data = {

            source:
                document.getElementById(
                    "source"
                ).value,

            street:
                document.getElementById(
                    "street"
                ).value,

            city:
                document.getElementById(
                    "city"
                ).value,

            county:
                document.getElementById(
                    "county"
                ).value,

            state:
                document.getElementById(
                    "state"
                ).value,

            zipcode:
                document.getElementById(
                    "zipcode"
                ).value,

            start_lat:
                Number(
                    document.getElementById(
                        "start_lat"
                    ).value
                ),

            start_lng:
                Number(
                    document.getElementById(
                        "start_lng"
                    ).value
                ),

            distance:
                Number(
                    document.getElementById(
                        "distance"
                    ).value
                ),

            start_time:
                document.getElementById(
                    "start_time"
                ).value,

            temperature:
                Number(
                    document.getElementById(
                        "temperature"
                    ).value
                ),

            wind_chill:
                optionalNumber(
                    "wind_chill"
                ),

            humidity:
                Number(
                    document.getElementById(
                        "humidity"
                    ).value
                ),

            pressure:
                Number(
                    document.getElementById(
                        "pressure"
                    ).value
                ),

            visibility:
                Number(
                    document.getElementById(
                        "visibility"
                    ).value
                ),

            wind_speed:
                optionalNumber(
                    "wind_speed"
                ),

            precipitation:
                optionalNumber(
                    "precipitation"
                ),

            wind_direction:
                optionalText(
                    "wind_direction"
                ),

            weather_condition:
                optionalText(
                    "weather_condition"
                ),

            airport_code:
                optionalText(
                    "airport_code"
                ),

            timezone:
                optionalText(
                    "timezone"
                ),

            sunrise_sunset:
                optionalText(
                    "sunrise_sunset"
                ),

            civil_twilight:
                optionalText(
                    "civil_twilight"
                ),

            nautical_twilight:
                optionalText(
                    "nautical_twilight"
                ),

            astronomical_twilight:
                optionalText(
                    "astronomical_twilight"
                )
        };


        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/predict",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                data
                            )
                    }
                );


            const prediction =
                await response.json();


            if (
                !response.ok ||
                !prediction.success
            ) {

                throw new Error(
                    prediction.error ||
                    "Prediction failed."
                );
            }


            showResult(
                prediction
            );


        } catch (error) {

            errorBox.textContent =
                error.message;

            errorBox.classList.remove(
                "hidden"
            );
        }

    }
);


function optionalNumber(id) {

    const value =
        document.getElementById(
            id
        ).value;

    if (
        value === "" ||
        value === null
    ) {

        return null;
    }

    return Number(value);
}


function optionalText(id) {

    const value =
        document.getElementById(
            id
        ).value
        .trim();

    if (value === "") {

        return null;
    }

    return value;
}


function showResult(
    prediction
) {

    severityResult.textContent =
        `Severity ${prediction.severity}`;


    descriptionResult.textContent =
        prediction.description;


    probabilities.innerHTML =
        "";


    for (
        const severity
        in prediction.probabilities
    ) {

        const probability =
            prediction.probabilities[
                severity
            ];


        const div =
            document.createElement(
                "div"
            );


        div.className =
            "probability";


        div.textContent =
            `Severity ${severity}: `
            +
            `${(
                probability * 100
            ).toFixed(2)}%`;


        probabilities.appendChild(
            div
        );
    }


    result.classList.remove(
        "hidden"
    );
}