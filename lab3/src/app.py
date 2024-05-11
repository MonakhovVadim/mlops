import streamlit as st

import model


@st.cache_resource
def get_model():
    return model.get_model()


def main():

    model, features, class_names = get_model()

    st.title("Введите параметры ириса ниже для определения его типа")

    elements = []
    for feature in features:
        element = st.number_input(
            feature, min_value=0.0, max_value=10.0, value=2.0, step=0.1
        )
        elements.append(element)

    if st.button("Определить тип ириса"):
        predict = model.predict([elements])[0]
        st.write(
            f"По указанным параметрам ирис определен как **{class_names[predict]}**"
        )


if __name__ == "__main__":
    main()
